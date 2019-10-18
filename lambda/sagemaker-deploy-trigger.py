import os
import json
import datetime
import boto3
import botocore
import tempfile
import logging
from boto3.session import Session

logger = logging.getLogger()
logger.setLevel(logging.INFO)

boto3.set_stream_logger(level=1)

codepipeline = boto3.client('codepipeline')
sagemaker = boto3.client('sagemaker')
dynamodb = boto3.client('dynamodb')


def main(event,context):
    """
    This function deploys the Inference Endpoint for the model generated at codepipeline execution runtime
    Calls to DynamoDB meta data store are also made to keep track of changes to environment in previous stage.

    :param event: is the event object that is passed when lambda function is triggered
    :param context: Another object passed to the environment
    :return: The function ends once a response has been given without the continuation token
    """
    job_id = event['CodePipeline.job']['id']
    job_data = event['CodePipeline.job']['data']
    try:
        input_artifact = job_data['inputArtifacts'][0]
        credentials = job_data['artifactCredentials']
        from_bucket = input_artifact['location']['s3Location']['bucketName']
        from_key = input_artifact['location']['s3Location']['objectKey']
        key_id = credentials['accessKeyId']
        key_secret = credentials['secretAccessKey']
        session_token = credentials['sessionToken']

        if "continuationToken" in job_data:
            continuation_token = job_data["continuationToken"]

            res = sagemaker.describe_endpoint(EndpointName=str(continuation_token))
            endpoint_service_status = str(res['EndpointStatus'])

            if endpoint_service_status == "Creating":
                msg = "Model Hosting Endpoint is being Created"
                print(msg)
                codepipeline.put_job_success_result(jobId=job_id, continuationToken=continuation_token)

            if endpoint_service_status == "Updating":
                msg = "Model Hosting Endpoint is being Updated"
                print(msg)
                codepipeline.put_job_success_result(jobId=job_id, continuationToken=continuation_token)

            if endpoint_service_status == "SystemUpdating":
                msg = "Model Hosting Endpoint System is being Updated"
                print(msg)
                codepipeline.put_job_success_result(jobId=job_id, continuationToken=continuation_token)

            if endpoint_service_status == "InService":
                msg = "Model Hosting Endpoint is now InService"
                print(msg)
                codepipeline.put_job_success_result(jobId=job_id)

            if endpoint_service_status == "Failed":
                msg = "Model Hosting Endpoint Creation Failed"
                print(msg)
                codepipeline.put_job_failure_result(jobId=job_id, failureDetails={'message': 'Endpoint Creation Failed',
                                                                                  'type': 'JobFailed'})
            if endpoint_service_status == "RollingBack":
                msg = "Model Hosting Endpoint Encountered Errors"
                print(msg)
                codepipeline.put_job_failure_result(jobId=job_id, failureDetails={'message': 'Endpoint Creation Rollback',
                                                                                  'type': 'JobFailed'})
            if endpoint_service_status == "OutOfService":
                msg = "Model Hosting Endpoint Creation Failed"
                print(msg)
                codepipeline.put_job_failure_result(jobId=job_id, failureDetails={'message': 'Endpoint Out of Service',
                                                                                  'type': 'JobFailed'})
        else:
            session = Session(aws_access_key_id=key_id, aws_secret_access_key=key_secret,
                              aws_session_token=session_token)
            s3 = session.client('s3', config=botocore.client.Config(signature_version='s3v4'))

            with tempfile.NamedTemporaryFile() as tmp:
                s3.download_file(from_bucket, from_key, tmp.name)
                with open(tmp.name) as f:
                    contents = f.readline()
                training_job_name = json.loads(contents)['training_job_name']

            res = sagemaker.list_training_jobs()
            for job in res['TrainingJobSummaries']:
                if job['TrainingJobName'] == str(training_job_name):
                    job_status = str(job['TrainingJobStatus'])
                    job_creation_time = str(job['CreationTime'])
                    job_end_time = str(job['TrainingEndTime'])

                    dynamodb.update_item(
                        TableName=str(os.environ['META_DATA_STORE']),
                        Key={'training_job_name': {'S': training_job_name}},
                        UpdateExpression="SET #job_creation_time= :val1, #job_end_time= :val2, #job_status= :val3",
                        ExpressionAttributeNames={'#job_creation_time': 'job_creation_time',
                                                  '#job_end_time': 'job_end_time',
                                                  '#job_status': 'job_status'
                                                  },
                        ExpressionAttributeValues={':val1': {'S': job_creation_time},
                                                   ':val2': {'S': job_end_time},
                                                   ':val3': {'S': job_status}
                                                   }
                    )

            inference_img_uri = str(os.environ['FULL_NAME'])
            endpoint_name = str(os.environ["IMG"]) + '-' + str(datetime.datetime.today()).replace(' ', '-').replace(':', '-').rsplit('.')[0]
            model_name = 'model-' + str(endpoint_name)
            endpoint_config_name = 'endpoint-config-'+ str(endpoint_name)
            model_obj_key = "{}/output/model.tar.gz".format(training_job_name)
            model_data_url = 's3://{}/{}'.format(str(os.environ['DEST_BKT']), model_obj_key)

            mod_res = sagemaker.create_model(ModelName=model_name, ExecutionRoleArn=str(os.environ['SAGE_ROLE_ARN']),
                                             PrimaryContainer={
                                                 'Image': inference_img_uri,
                                                 'ModelDataUrl': model_data_url
                                             })
            print(mod_res)
            conf_res = sagemaker.create_endpoint_config(EndpointConfigName=endpoint_config_name, ProductionVariants=[{
                'VariantName': 'initial-variant',
                'ModelName': model_name,
                'InitialInstanceCount': 1,
                'InstanceType': 'ml.t2.medium'
            }])
            print(conf_res)
            endpoint_res = sagemaker.create_endpoint(EndpointName=endpoint_name,
                                                     EndpointConfigName=endpoint_config_name
                                                     )
            print(endpoint_res)

            dynamodb.update_item(
                TableName=str(os.environ['META_DATA_STORE']),
                Key={'training_job_name': {'S': training_job_name}},
                UpdateExpression="SET #inference_image_uri= :val3, #model_name= :val4, #endpoint_config_name= :val5, " +
                                 "#endpoint_name= :val6",
                ExpressionAttributeNames={
                                          '#inference_image_uri': 'inference_image_uri',
                                          '#model_name': 'model_name',
                                          '#endpoint_config_name': 'endpoint_config_name',
                                          '#endpoint_name': 'endpoint_name'
                                          },
                ExpressionAttributeValues={
                                           ':val3': {'S': inference_img_uri},
                                           ':val4': {'S': model_name},
                                           ':val5': {'S': endpoint_config_name},
                                           ':val6': {'S': endpoint_name}
                                           }
            )
            if 'EndpointArn' in endpoint_res.keys():
                codepipeline.put_job_success_result(jobId=job_id, continuationToken=endpoint_name)
            else:
                codepipeline.put_job_failure_result(jobId=job_id, failureDetails={'message': 'Endpoint not Created',
                                                                                  'type': 'JobFailed'})
    except Exception as e:
        print(e)
        codepipeline.put_job_failure_result(jobId=job_id, failureDetails={'message': str(e), 'type': 'JobFailed'})