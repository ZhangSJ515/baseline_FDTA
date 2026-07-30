from __future__ import annotations

import torch

from models.misc import get_model
from models.runtime_tracker import RuntimeTracker
from utils.box_ops import box_cxcywh_to_xywh
from utils.misc import distributed_device


class AirMOTRuntimeTracker(RuntimeTracker):
    """FDTA runtime tracker with optional class-consistent association.

    The original FDTA benchmarks are single-class datasets. AirMOT contains
    four heterogeneous classes, so an ID candidate from a different class must
    not be selected merely because its embedding score is high. This subclass
    masks cross-class ID scores and keeps the category of an established track
    stable over time. The underlying FDTA association model is unchanged.
    """

    def __init__(self, *args, class_aware: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_aware = class_aware
        self.id_label_to_category: dict[int, int] = {}

    @torch.no_grad()
    def update(self, image):
        detr_out = self.model(frames=image, part="detr")
        scores, categories, boxes, output_embeds = self._get_activate_detections(
            detr_out=detr_out
        )

        if self.only_detr:
            id_pred_labels = self.num_id_vocabulary * torch.ones(
                boxes.shape[0], dtype=torch.int64, device=boxes.device
            )
        else:
            id_pred_labels = self._get_id_pred_labels(
                boxes=boxes,
                output_embeds=output_embeds,
                categories=categories,
            )

        # Remove low-confidence newborn detections while retaining detections
        # that are associated with an existing identity.
        keep_idxs = (
            (id_pred_labels != self.num_id_vocabulary)
            | (scores > self.newborn_thresh)
        )
        scores = scores[keep_idxs]
        categories = categories[keep_idxs]
        boxes = boxes[keep_idxs]
        output_embeds = output_embeds[keep_idxs]
        id_pred_labels = id_pred_labels[keep_idxs]

        n_activate_id_labels = 0
        n_newborn_targets = 0
        for id_label in id_pred_labels.tolist():
            if id_label != self.num_id_vocabulary:
                n_activate_id_labels += 1
                self.id_queue.add(id_label)
            else:
                n_newborn_targets += 1

        n_remaining_ids = len(self.id_queue) - n_activate_id_labels
        if n_newborn_targets > n_remaining_ids:
            keep_idxs = torch.ones(
                len(id_pred_labels), dtype=torch.bool, device=id_pred_labels.device
            )
            newborn_idxs = id_pred_labels == self.num_id_vocabulary
            newborn_keep_idxs = torch.ones(
                int(newborn_idxs.sum().item()),
                dtype=torch.bool,
                device=id_pred_labels.device,
            )
            newborn_keep_idxs[n_remaining_ids:] = False
            keep_idxs[newborn_idxs] = newborn_keep_idxs
            scores = scores[keep_idxs]
            categories = categories[keep_idxs]
            boxes = boxes[keep_idxs]
            output_embeds = output_embeds[keep_idxs]
            id_pred_labels = id_pred_labels[keep_idxs]

        newborn_mask = id_pred_labels == self.num_id_vocabulary
        id_labels = self._assign_newborn_id_labels(pred_id_labels=id_pred_labels)

        if len(torch.unique(id_labels)) != len(id_labels):
            raise RuntimeError(
                f"Duplicate ID labels were assigned in one frame: {id_labels.tolist()}"
            )

        # Lock the semantic class after a track is born. This prevents a
        # transient class prediction from changing the class used for output
        # and for future class-aware association.
        stable_categories = categories.clone()
        for index, id_label in enumerate(id_labels.tolist()):
            detected_category = int(categories[index].item())
            if bool(newborn_mask[index].item()) or id_label not in self.id_label_to_category:
                self.id_label_to_category[id_label] = detected_category
            elif self.class_aware:
                stable_categories[index] = self.id_label_to_category[id_label]

        self.current_track_results = {
            "score": scores,
            "category": stable_categories,
            "bbox": box_cxcywh_to_xywh(boxes) * self.bbox_unnorm,
            "id": torch.tensor(
                [self.id_label_to_id[label] for label in id_labels.tolist()],
                dtype=torch.int64,
            ),
        }

        for id_label in id_labels.tolist():
            self.id_queue.add(id_label)

        self._update_trajectory_infos(
            boxes=boxes,
            output_embeds=output_embeds,
            id_labels=id_labels,
        )
        self._filter_out_inactive_tracks()

        # Remove category entries for ID-vocabulary slots that are no longer
        # represented by an active trajectory. A recycled slot will receive
        # the category of its new track at birth.
        if self.trajectory_id_labels.shape[0] > 0:
            active_labels = set(self.trajectory_id_labels[0].tolist())
        else:
            active_labels = set()
        self.id_label_to_category = {
            label: category
            for label, category in self.id_label_to_category.items()
            if label in active_labels
        }

    def _get_id_pred_labels(
            self,
            boxes: torch.Tensor,
            output_embeds: torch.Tensor,
            categories: torch.Tensor,
    ):
        if boxes.shape[0] == 0:
            return torch.zeros((0,), dtype=torch.int64, device=boxes.device)

        if self.trajectory_features.shape[0] == 0:
            return self.num_id_vocabulary * torch.ones(
                boxes.shape[0], dtype=torch.int64, device=boxes.device
            )

        current_features = output_embeds[None, ...]
        current_boxes = boxes[None, ...]
        current_masks = torch.zeros(
            (1, output_embeds.shape[0]),
            dtype=torch.bool,
            device=distributed_device(),
        )
        current_times = self.trajectory_times.shape[0] * torch.ones(
            (1, output_embeds.shape[0]),
            dtype=torch.int64,
            device=distributed_device(),
        )
        seq_info = {
            "trajectory_features": self.trajectory_features[None, None, ...],
            "trajectory_boxes": self.trajectory_boxes[None, None, ...],
            "trajectory_id_labels": self.trajectory_id_labels[None, None, ...],
            "trajectory_times": self.trajectory_times[None, None, ...],
            "trajectory_masks": self.trajectory_masks[None, None, ...],
            "unknown_features": current_features[None, None, ...],
            "unknown_boxes": current_boxes[None, None, ...],
            "unknown_masks": current_masks[None, None, ...],
            "unknown_times": current_times[None, None, ...],
        }
        seq_info = self.model(seq_info=seq_info, part="trajectory_embedding")
        seq_info = self.model(seq_info=seq_info, part="trajectory_modeling")
        id_logits, _, _ = self.model(seq_info=seq_info, part="id_decoder")
        id_logits = id_logits[0, 0, 0]

        if not self.use_sigmoid:
            id_scores = id_logits.softmax(dim=-1)
        else:
            id_scores = id_logits.sigmoid()

        if self.class_aware:
            active_id_labels = set(self.trajectory_id_labels[0].tolist())
            for object_index, category in enumerate(categories.tolist()):
                for id_label in active_id_labels:
                    track_category = self.id_label_to_category.get(id_label)
                    if track_category is not None and track_category != int(category):
                        id_scores[object_index, id_label] = 0.0

        match self.assignment_protocol:
            case "hungarian":
                id_labels = self._hungarian_assignment(id_scores=id_scores)
            case "object-max":
                id_labels = self._object_max_assignment(id_scores=id_scores)
            case "id-max":
                id_labels = self._id_max_assignment(id_scores=id_scores)
            case _:
                raise NotImplementedError(
                    f"Unsupported assignment protocol: {self.assignment_protocol}"
                )

        return torch.tensor(
            id_labels,
            dtype=torch.int64,
            device=distributed_device(),
        )
