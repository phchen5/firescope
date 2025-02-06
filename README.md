# 🔥 Fire Scope: Fire Image Detection with CNN and Transfer Learning (98% acc)¶

In this project, we'll be building image classification models using CNN and transfer learning. The analysis can be accessed [here](analysis/firescope-cnn-and-tl-with-pytorch-98-acc.ipynb).

**Author**: Po-Hsun (Ben) Chen

## Summary

In this project, we have a dataset that contains two types of images: fire images and non-fire images. Our objective is to be able to classify these images into their corresponding categories. To complete this task, we'll be building two of our own convolutional neural network (CNN), a simple one and a slightly more complicated one. At the very end, we'll compare the performance of our model to a pretrained model and see for ourselves the incredible power of transfer learning.

## Data Source

The dataset used in this analysis consists of 2 folders, each containing fire and non-fire images. In total, we have 755 outdoor fire images and 244 non-fire nature images. I've also gathered three random images from the internet to test the final capability of our model. The dataset is sourced from Kaggle and you can access the dataset [here](https://www.kaggle.com/datasets/phylake1337/fire-dataset). The three images sourced from the internet for testing the model can be accssed [here](https://www.kaggle.com/datasets/phchen5/custom-fire-images)

## Deployment

TThe model is deployed using Streamlit Cloud for easy access. You can try it out here: [FireScope on Streamlit](https://firescope.streamlit.app/).

To scale the application and handle higher traffic, the model is also deployed on AWS ECS using Fargate. You can access the AWS deployment here: [FireScope on AWS ECS](http://3.143.210.210:8501/).

To run the application on your local machine, follow these steps:

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run app.py
```

## Dependencies

All required dependencies are listed in this [conda environment file](environment.yaml).

## How to Reproduce the Analysis

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/phchen5/firescope.git
   cd firescope
   ```

2. **Download the Dataset from the Source:**
   You may need to register for a Kaggle account. After you've downloaded the dataset, place the `fire_images/` and `non_fire_images/` folder within a `data/` folder located in the root. The three images for testing should also be placed in another `test_images/` folder located in the root. You can choose to place these anywhere else in the repository as you like, but just be sure to edit the data source path within the analysis.

3. **Set Up and Activate Environment:**

   ```bash
   conda env create -f environment.yaml
   conda activate firescope
   ```

4. **Open the Notebook:**

   ```bash
   jupyter lab analysis/firescope-cnn-and-tl-with-pytorch-98-acc.ipynb
   ```

5. **Run the Cells and Have Fun Exploring!**

## Files

- `analysis/firescope-cnn-and-tl-with-pytorch-98-acc.ipynb`: Jupyter notebook containing the EDA code.
- `environment.yaml`: Conda environment file listing required dependencies.
- `deployment/best_baseline_model.pth`: The saved model weights for the baseline CNN model.
- `deployment/best_model_1.pth`: The saved model weights for the slightly more complicated CNN model.
- `deployment/best_model_tl.pth`: The saved model weights for the model trained using transfer learning.
- `deployment/app.py`: The app file for deployment on Streamlit.
- `deployment/requirements.txt`: The file that contains the required dependencies for deployment on Streamlit.

