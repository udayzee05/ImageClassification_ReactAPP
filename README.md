#  to run Docker tensorflow serving command
docker run -it -v E:\Research\ReactAPP:/reactapp -p 8601:8601 --entrypoint /bin/bash tensorflow/serving

# docker tensorflow model
tensorflow_model_server --rest_api_port=8601 --model_name=potato_model --model_base_path=/reactapp/saved_models/

# GCP run Command
gcloud functions deploy predict --runtime python38 --trigger-http --memory 512 --project lyrical-carver-377818