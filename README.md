## Number Plate Detection using Attention OCR

## Problem

We will use tensorflow attention-ocr to predict the text present on number plates. 

## Usage

Please make sure you have the latest versions of tensorflow, opencv and pandas installed. 

## Approach
Getting training data
We have images of number plates but we do not have the text in them or the bounding box numbers of the number plates in these images. Use an annotation tool to get your annotations and save them in a .csv file.


###Getting crops
We have stored our bounding box data as a .csv file. The .csv file has the following fields:

files
text
xmin
xmax
ymin
ymax

To crop the images and get only the cropped window we have to deal with different sized images. To do this we read the csv data in as a pandas dataframe and get our coordinates in such a way that we don't miss any information about the number plates while also maintaining a constant size of the crops. This will prove helpful when we are training our OCR model.

script present in ```bash get_crops.py```

###Generate tfrecords
Having stored our cropped images of equal sizes in a different directory, we can begin using those images to generate tfrecords that we will use to train our dataset. Here's a script to generate tfrecords. Note the max_width and max_height variables so we can specify the size of our crops to our tfrecord generation script. These tfrecords along with the label mapping have to be stored in the tensorflow object detection API inside the following directory -

The dataset has to be in the FSNS dataset format. For this, your test and train tfrecords along with the charset labels text file are placed inside a folder named 'fsns' inside the 'datasets' directory. you can change this to another folder and upload your tfrecord files and charset-labels.txt here. You'll have to change the path in multiple places accordingly.

tfrecord generation script present in 'get_tf_records.py'

###Setting our Attention-OCR up
Once we have our tfrecords and charset labels stored in the required directory, we need to write a dataset config script that will help us split our data into train and test for the attention OCR training script to process.

Make a python file and name it 'number_plates.py' and place it inside the following directory:

'models/research/attention_ocr/python/datasets'
The contents of the number-plates.py can be seen [here](https://github.com/codeaway23/models/blob/master/research/attention_ocr/python/datasets/number_plates.py). 

###Training the model
Move into the following directory: models/research/attention_ocr
Open the file named 'common_flags.py' and specify where you'd want to log your training.

then run
```bash
python train.py --dataset_name=number_plates --max_number_of_steps=3000
```
###Evaluating the model
Run the following command from terminal.

```bash
python eval.py --dataset_name='number_plates'
```

Get predictions
In models/research/attention_ocr/python open the file demo_inference.py and comment out the following lines.

```python
def load_images(file_pattern, batch_size, dataset_name):
  width, height = get_dataset_image_size(dataset_name)
  images_actual_data = np.ndarray(shape=(batch_size, height, width, 3),
                                  dtype='uint8')
  for i in range(batch_size):
    # path = file_pattern % i
    # print("Reading %s" % path)
    pil_image = PIL.Image.open(tf.gfile.GFile(file_pattern))
    images_actual_data[i, ...] = np.asarray(pil_image)
  return images_actual_data
```

Now from the same directory run the following command on your shell.

```bash
python demo_inference.py --dataset_name=number_plates \n 
--batch_size=1, checkpoint='models/research/attention_ocr/number_plates_model_logs/model.ckpt-3000', \n
--image_path_pattern=/home/anuj/crops/111.png
```
