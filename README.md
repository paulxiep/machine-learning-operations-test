# machine-learning-operations-test
A test for mid level Machine learninng operations (MLOps) for a company in Thailand

### Instruction

A _junior_ data scientist has prepared a prediction model ready to predict client's orders for a restaurant - given that a scientist has no prior knowledge on machine learning operations and bringing the models up for production, you, as a machine learning engineer, has to bring this model to live by yourself. Please *demonstrate* how would you do it. 

Fork this repo and make this model ready to deploy on GCP on any suitable service of your choice


### Methods

1. Streamlit: Simple, easy to use, web application tool suitable for small data application.
2. Airflow Docker: While normally associated with data pipeline, Airflow can be adapted to schedule regular model predict/retraining service.

### How to run

0. All methods: ```sh prepare_demo.sh``` to prepare dummy models and synthetic data for demo runs on all methods.
1. Streamlit: ```streamlit run simple_streamlit.py```, then insert test data with categorical features already encoded (can be obtained from ```/data``` after preparing demo in step 0). The results will be displayed on the streamlit web app.
2. Airflow Docker: ```sh prepare_airflow_image.sh```, then ```docker-compose up```. Go to ```localhost:8080``` on your browser (requires said port to be vacant), and login with user=(first word of company name), password=(last word of company name). Click to activate the daily_prediction_run DAG manually. The prediction results and generated test data will be in ```/results``` folder. When done run ```docker-compose down``` and clean out unneeded docker images.

### Mid-Development Updates Log

2023-11-17 21:53: Added data synthesizer to somewhat mimic the data in jupyter notebook.

2023-11-17 22:18: Added module to mimic the preprocessing in the jupyter notebook.

2023-11-17 22.40: Added 2nd part of preprocess for feature set 2.

2023-11-18 01:06: Prepared dummy model modules and model saving functionalities (Stacking doesn't work yet)

2023-11-18 13:35: Method 1 (Streamlit) up (still without Stacking model until further notice)

2023-11-18 16:20: Method 2 (Airflow Docker) up (with only prediction DAG).

2023-11-19 11:15: Restructured project in preparation for Method 3.
