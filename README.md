Getting Started
1. Prerequisites
Ensure you have Conda installed. Use the provided environment.yml file to set up the environment:

bash
复制代码
conda env create -f environment.yml
conda activate srresnet
2. Dataset Preparation
Place the training datasets in the trainset/ folder.
Place the testing datasets in the testset/ folder.
3. Training the Model
Run the training script using the provided shell script:

bash
复制代码
bash train.sh
Alternatively, you can execute the Python script directly:

bash
复制代码
python main_train_psnr.py
4. Testing the Model
Use the test script to evaluate the model on the test dataset:

bash
复制代码
bash test.sh
5. Clean Prior Generation
Navigate to the clean_prior_generate/MAE/ folder and follow the instructions in MAE_prior.ipynb for generating clean priors.
