# Training/Inference with Image Modality
If your training and inference tasks involve image modality, your data would contain a column representing the path to the image file, i.e.

```python
   feature_1                     image   label
0          1   image/train/train_1.png       0
1          2   image/train/train_1.png       1
```

### Preparing the Image Column
Currently, AutoGluon-Cloud only supports one image per row.
If your dataset contains one or more images per row, we first need to preprocess the image column to only contain the first image of each row.

For example, if your images are seperated with `;`, you can preprocess it via:

```python
# image_col is the column name containing the image path. In the example above, it would be `image`
train_data[image_col] = train_data[image_col].apply(lambda ele: ele.split(';')[0])
test_data[image_col] = test_data[image_col].apply(lambda ele: ele.split(';')[0])
```

Now we update the path to an absolute path.

For example, if your directory is similar to this:

```bash
.
└── current_working_directory/
    ├── train.csv
    ├── test.csv
    └── images/
        ├── train/
        │   └── train_1.png
        └── test/
            └── test_1.png
```

You can replace your image column to absolute paths via:

```python
train_data[image_col] = train_data[image_col].apply(lambda path: os.path.abspath(path))
test_data[image_col] = test_data[image_col].apply(lambda path: os.path.abspath(path))
```

### Perform Training/Inference with Image Modality
Provide argument `image_column` as the column name containing image paths to `CloudPredictor` fit/inference APIs along with other arguments that you would normally pass to a CloudPredictor
In the example above, `image_column` would be `image`

```python
cloud_predictor = TabularCloudPredictor(cloud_output_path="YOUR_S3_BUCKET_PATH")
cloud_predictor.fit(..., image_column="IMAGE_COLUMN_NAME")
cloud_predictor.predict_real_time(..., image_column="IMAGE_COLUMN_NAME")
cloud_predictor.predict(..., image_column="IMAGE_COLUMN_NAME")
```
