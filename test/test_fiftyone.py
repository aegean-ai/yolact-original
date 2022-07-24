



# # A name for the dataset
# name = "NJTPA"

# # The directory containing the dataset to import
# data_path = "/workspaces/data/njtpa.auraison.aegean.ai/dvrpc-pedestrian-network-pa-only-2020/Test/*"
# labels_path = "/workspaces/data/njtpa.auraison.aegean.ai/dvrpc-pedestrian-network-pa-only-2020/DVRPC_test.json"

# # The type of the dataset being imported
# dataset_type = fo.types.COCODetectionDataset

# dataset = fo.Dataset.from_dir(
#     dataset_type=dataset_type,
#     data_path=data_path,
#     labels_path=labels_path,
#     name=name,
# )

# import glob
# import fiftyone as fo

# images_path = "/workspaces/data/njtpa.auraison.aegean.ai/dvrpc-pedestrian-network-pa-only-2020/Test/"

# # Ex: your custom label format
# annotations = {
#     "/path/to/images/000001.jpg": [
#         {"bbox": ..., "label": ...},
#         ...
#     ],
#     ...
# }

# # Create samples for your data
# samples = []
# for filepath in glob.glob(images_path):
#     sample = fo.Sample(filepath=filepath)

#     # Convert detections to FiftyOne format
#     detections = []
#     for obj in annotations[filepath]:
#         label = obj["label"]

#         # Bounding box coordinates should be relative values
#         # in [0, 1] in the following format:
#         # [top-left-x, top-left-y, width, height]
#         bounding_box = obj["bbox"]

#         detections.append(
#             fo.Detection(label=label, bounding_box=bounding_box)
#         )

#     # Store detections in a field name of your choice
#     sample["ground_truth"] = fo.Detections(detections=detections)

#     samples.append(sample)

# # Create dataset
# dataset = fo.Dataset("my-detection-dataset")
# dataset.add_samples(samples)