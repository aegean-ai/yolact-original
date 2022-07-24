import fiftyone.utils.coco as fouc

#
# Mock COCO predictions, where:
# - `image_id` corresponds to the `coco_id` field of `coco_dataset`
# - `category_id` corresponds to classes in `coco_dataset.default_classes`
#
predictions = [
    {"image_id": 1, "category_id": 18, "bbox": [258, 41, 348, 243], "score": 0.87},
    {"image_id": 2, "category_id": 11, "bbox": [61, 22, 504, 609], "score": 0.95},
]

# Add COCO predictions to `predictions` field of dataset
fouc.add_coco_labels(coco_dataset, "predictions", predictions)

# Verify that predictions were added to two images
print(coco_dataset.count("predictions"))  # 2