FROM apache/airflow:3.0.0

# Installation des dépendances Python pour THORAXIA pipeline
USER airflow
RUN pip install --no-cache-dir \
    kaggle==1.6.17 \
    pydicom==2.4.4 \
    pillow==10.4.0 \
    tqdm==4.66.5 \
    pandas==2.2.2 \
    numpy==1.26.4