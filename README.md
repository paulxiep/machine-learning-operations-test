# machine-learning-operations-test
A test for mid level Machine learninng operations (MLOps) for a company in Thailand

### Instruction

A _junior_ data scientist has prepared a prediction model ready to predict client's orders for a restaurant - given that a scientist has no prior knowledge on machine learning operations and bringing the models up for production, you, as a machine learning engineer, has to bring this model to live by yourself. Please *demonstrate* how would you do it. 

Fork this repo and make this model ready to deploy on GCP on any suitable service of your choice.

### Important

Will require ```.env``` file with corresponding keys to run.

### Disclaimer

1. While there were attempts to ensure compatibilities across platforms, the main code was only tested on Windows.
2. As I only trained models from generated data, I didn't include the code for data cleaning, outlier removal, or data exploration.
3. Most of the work here was focused on model deployment as a service, not on model retraining nor the whole of MLOps itself.

### Methods

1. Streamlit: Simple, easy to use, web application tool suitable for small data application.
2. Airflow Docker: While normally associated with data pipeline, Airflow can be adapted to schedule regular model predict/retraining service. This can be used in conjunction with DVC to manage model and data versioning.
3. Flask API on Docker: I once used the company's own application in place of Flask to enable API on cloud. I'm a newbie to Flask itself and may not be familiar with security protocol. Nevertheless I made a working Flask API application, and can be tested with your own API call program or my streamlit program. This can be refined further as I get more familiar with Flask API.
4. Kubernetes (not implemented): Once there's a packaged API software on Docker, one can attempt to deploy it on cloud using orchestrator and load manager like Kubernetes. Where I once worked at this duty fell to a Backend Developer and admittedly I didn't get to do it. I won't shy away if the duty falls to me but for the time being it is not yet my expertise.

### How to run

0. All methods: ```sh 00_prepare_demo.sh``` to prepare dummy models and synthetic data for demo runs on all methods.
1. Streamlit: ```streamlit run simple_streamlit.py```, then insert test data with categorical features already encoded (can be obtained from ```/data``` after preparing demo in step 0). The results will be displayed on the streamlit web app. Don't touch the radio button yet.
2. Airflow Docker: (requires running docker engine) ```sh 02_prepare_airflow_demo.sh```. Go to ```localhost:{AIRFLOW_PORT}``` on your browser (requires said port to be vacant), and login with user and password specified in .env. Click to enable the daily_training_run and daily_prediction_run DAG and wait for scheduled run or activate more runs manually. Newly trained models will be in ```/models``` folder. The prediction results and generated test data will be in ```/results``` folder. When done run ```sh 10_clean_docker.sh```.
3. Flask API on Docker: (requires running docker engine) ```sh 03_prepare_flask_demo.sh```. Then run ```streamlit run simple_streamlit.py```, except this time choose 'Call Flask API' on the radio button, then uploaded test data. When done run ```sh 10_clean_docker.sh```.

### Mid-Development Updates Log

2023-11-17 21:53: Added data synthesizer to somewhat mimic the data in jupyter notebook.

2023-11-17 22:18: Added module to mimic the preprocessing in the jupyter notebook.

2023-11-17 22.40: Added 2nd part of preprocess for feature set 2.

2023-11-18 01:06: Prepared dummy model modules and model saving functionalities (Stacking doesn't work yet)

2023-11-18 13:35: Method 1 (Streamlit) up (still without Stacking model until further notice)

2023-11-18 16:20: Method 2 (Airflow Docker) up (with only prediction DAG).

2023-11-19 11:15: Restructured project in preparation for Method 3.

2023-11-19 16:10: Method 3 (Flask Web API) up and simplified running steps.

2023-11-20 12:30: Method 2 training DAG up and final notes before submission.
