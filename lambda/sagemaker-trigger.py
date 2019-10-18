import os
import json
import datetime
import boto3
import botocore
import zipfile
import tempfile

from boto3.session import Session

boto3.set_stream_logger(level=1)
ArtifactFileName = "outfile.txt"
codepipeline = boto3.client('codepipeline')
sagemaker = boto3.client('sagemaker')
dynamodb = boto3.client('dynamodb')


def get_artifact(s3, bucketName, objectKey):
    """
    Downloads codepipeline artifact being passed into this stage/action.

    :param s3: client object (initialized with temp credentials)
    :param bucketName: name of S3 bucket
    :param objectKey: S3 object key name
    :return: returns the unzipped object passed as codepipeline artifact

    """
    with tempfile.NamedTemporaryFile() as tmp_file:
        s3.download_file(bucketName, objectKey, tmp_file.name)
        with zipfile.ZipFile(tmp_file.name, 'r') as zip:
            return zip.read(ArtifactFileName)


def create_data_config(bucket_uri, bucket_name, s3):
    """
    This function takes in the TrainingInputBucket name for this particular and returns the relevant inputDataConfig
    param of the sagemaker CreateTrainingJob API
    :param bucket: str
    :return: list of dicts
    """

    input_data_config = list()
    for ii in ['training', 'testing', 'validation']:
        resp = s3.list_objects(Bucket=bucket_name, Prefix='input/data/{}/'.format(ii))
        if resp['Contents']:
            for k in resp['Contents']:
                if k['Size'] > 0:
                    input_data_config.append(
                        {
                            'ChannelName': str(ii),
                            'DataSource': {
                                'S3DataSource': {
                                    'S3DataType': 'S3Prefix',
                                    'S3Uri': str(bucket_uri) + str(ii) + "/"
                                }
                            }
                        }
                    )
    return input_data_config


def main(event, context):
    """
    This function creates the sagemaker training job at the codepipeline execution runtime.
    It also writes relevant attributes of the pipeline execution to the DynamoDB meta data store

    :param event: Lambda event dict
    :param context: lamnda context object defined in the environment
    :return: The function stops when codepipeline is sent a response without the continuation token

    """

    job_id = event['CodePipeline.job']['id']
    job_data = event['CodePipeline.job']['data']
    try:
        input_artifact = job_data['inputArtifacts'][0]
        from_bucket = input_artifact['location']['s3Location']['bucketName']
        from_key = input_artifact['location']['s3Location']['objectKey']

        output_artifact = job_data['outputArtifacts'][0]
        to_bucket = output_artifact['location']['s3Location']['bucketName']
        to_key = output_artifact['location']['s3Location']['objectKey']

        credentials = job_data['artifactCredentials']
        key_id = credentials['accessKeyId']
        key_secret = credentials['secretAccessKey']
        session_token = credentials['sessionToken']

        if "continuationToken" in job_data:
            ## collect the value of the continuation token and describe sagemaker training job
            continuation_token = job_data["continuationToken"]

            res = sagemaker.describe_training_job(TrainingJobName=str(continuation_token))
            training_job_status = str(res["TrainingJobStatus"])

            if training_job_status == "InProgress":
                msg = "SageMaker Training Job In Progress"
                print(msg)
                codepipeline.put_job_success_result(jobId=job_id, continuationToken=continuation_token)

            if training_job_status == "Completed":
                msg = "SageMaker Training Job Completed"
                print(msg)
                codepipeline.put_job_success_result(jobId=job_id)

            if training_job_status == "Failed":
                msg = "SageMaker Training Job Failed"
                print(msg)
                codepipeline.put_job_failure_result(jobId=job_id, failureDetails={'message': 'SageMaker Job Failed',
                                                                                  'type': 'JobFailed'})

            if training_job_status == "Stopping":
                msg = "SageMaker Training Job Stopping"
                print(msg)
                codepipeline.put_job_success_result(jobId=job_id, continuationToken=continuation_token)

            if training_job_status == "Stopped":
                msg = "SageMaker Training Job Stopped"
                print(msg)
                codepipeline.put_job_failure_result(jobId=job_id, failureDetails={'message': 'SageMaker Job Stopped',
                                                                                  'type': 'JobFailed'})

        else:
            s3 = boto3.client('s3')

            response5 = s3.list_objects(Bucket=str(os.environ["SRC_BKT_NAME"]), Marker='input/config/', Prefix='input/config/')
            hyper_param_dict = {"foo": "bar"}
            if 'Contents' in response5:
                object_list5 = list()
                for key in response5['Contents']:
                    object_list5.append(key['Key'])
                if len(object_list5) >= 1:
                    print("Hyperparameters Already Exists")
                    result = s3.get_object(Bucket=str(os.environ["SRC_BKT_NAME"]), Key=object_list5[0])
                    text = result["Body"].read().decode()
                    hyper_param_dict = json.loads(text)

            input_hyperparams = dict()
            for key in hyper_param_dict.keys():
                dict_entry = {'S': str(hyper_param_dict[key])}
                input_hyperparams.update({str(key): dict_entry})


            input_data_config = create_data_config(
                bucket_uri=str(os.environ["SRC_BKT_URI"]),
                bucket_name=str(os.environ['SRC_BKT_NAME']),
                s3=s3
            )

            s3_key_version_dict = dict()
            versions = s3.list_object_versions(Bucket=str(os.environ["SRC_BKT_NAME"]), Prefix='input/')
            x = versions.get('Versions')
            for key in x:
                Key = key['Key']
                VersionId = key['VersionId']
                Size = key['Size']
                IsLatest = key['IsLatest']
                if Size != 0:
                    if IsLatest == True:
                        key_name = str(Key)
                        version_id = {'S': str(VersionId)}
                        s3_key_version_dict.update({key_name: version_id})

            session = Session(aws_access_key_id=key_id, aws_secret_access_key=key_secret,aws_session_token=session_token)
            s3 = session.client('s3', config=botocore.client.Config(signature_version='s3v4'))

            docs = get_artifact(s3, from_bucket, from_key)
            if (docs):
                docs_dict = json.loads(docs.decode("utf-8"))
                git_hash = docs_dict['COMMIT_ID']

            training_job_name = str(os.environ["IMG"]) + '-' + str(datetime.datetime.today()).replace(' ', '-').replace(':', '-').rsplit('.')[0]

            s3.put_object(Body=json.dumps({"training_job_name": training_job_name}), Bucket=to_bucket, Key=to_key)

            dynamodb.put_item(
                TableName=str(os.environ["META_DATA_STORE"]),
                Item={
                    'training_job_name': {'S': training_job_name},
                    'instance_type': {'S': str(os.environ["INSTANCE_TYPE"])},
                    'instance_count': {'N': str(os.environ["INSTANCE_CNT"])},
                    'instance_vol_in_gigs': {'N': str(os.environ["EBS_VOL_GB"])},
                    'git_hash': {'S': str(git_hash)},
                    'training_image_uri': {'S': str(os.environ["FULL_NAME"]) },
                    'input_bucket_name': {'S': str(os.environ["SRC_BKT_NAME"])},
                    'input_key_versions': {'M': s3_key_version_dict},
                    'input_hyperparms': {'M': input_hyperparams}
                }
            )

            hyper_param_dict.update({'meta_data_store': str(os.environ["META_DATA_STORE"])})

            sage_res = sagemaker.create_training_job(
                TrainingJobName=training_job_name,
                HyperParameters=hyper_param_dict,
                AlgorithmSpecification={
                    'TrainingImage': str(os.environ["FULL_NAME"]),
                    'TrainingInputMode': 'File'
                },
                RoleArn=str(os.environ["SAGE_ROLE_ARN"]),
                InputDataConfig=input_data_config,
                ResourceConfig={
                    'InstanceType': str(os.environ["INSTANCE_TYPE"]),
                    'InstanceCount': int(os.environ["INSTANCE_CNT"]),
                    'VolumeSizeInGB': int(os.environ["EBS_VOL_GB"])
                },
                OutputDataConfig={'S3OutputPath': str(os.environ["DEST_BKT_URI"])},
                StoppingCondition={'MaxRuntimeInSeconds': int(os.environ["RUN_TIME_SEC"])}
            )

            if 'TrainingJobArn' in sage_res.keys():
                codepipeline.put_job_success_result(jobId=job_id, continuationToken=training_job_name)
            else:
                codepipeline.put_job_failure_result(jobId=job_id, failureDetails={'message': 'Invalid Request',
                                                                                  'type': 'JobFailed'})
    except Exception as e:
        codepipeline.put_job_failure_result(jobId=job_id, failureDetails={'message': str(e), 'type': 'JobFailed'})