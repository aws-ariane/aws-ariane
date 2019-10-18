#!/bin/bash
AWS_DEFAULT_REGION="eu-central-1"
Email="ankush.basavarajappa@lhind.dlh.com"

# Parameters to Configure Specific Github Repo
GitHub_User="aws-ariane"
GitHub_Repo="aws-ariane"
GitHub_Branch="master"
GitHub_Token="b323f7d01152ede7cd85d36aa6453277a605e2dd"

# CodeBuild Project Parameters
Python_Build_Version="aws/codebuild/python:3.6.5-1.3.2"
Build_Timeout_Mins=30

# SageMaker Training Job Parameters
Instance_Count=1
Instance_Type="ml.m4.xlarge"
Max_Runtime_In_Seconds=86400
Vol_In_GB=60

Template_Name="${GitHub_Repo}-pipeline1"
Lambdas_Bucket="${Template_Name}-lambdas-`date '+%Y-%m-%d-%H-%M-%S'`"
Lambdas_Key="SageMakerTriggers/LambdaFunction.zip"

cd lambda

chmod -R 755 .

zip -r ../LambdaFunction.zip .

cd ..

aws s3api create-bucket --bucket ${Lambdas_Bucket} --create-bucket-configuration LocationConstraint=eu-central-1

aws s3api put-object --bucket ${Lambdas_Bucket} \
  --key ${Lambdas_Key} \
  --body ./LambdaFunction.zip

aws cloudformation create-stack \
  --region ${AWS_DEFAULT_REGION} \
  --stack-name ${Template_Name} \
  --template-body file://template/sagemaker-pipeline.yaml \
  --parameters \
    ParameterKey=LambdasBucket,ParameterValue=${Lambdas_Bucket} \
    ParameterKey=LambdasKey,ParameterValue=${Lambdas_Key} \
    ParameterKey=Email,ParameterValue=${Email} \
    ParameterKey=GitHubUser,ParameterValue=${GitHub_User} \
    ParameterKey=GitHubRepo,ParameterValue=${GitHub_Repo} \
  	ParameterKey=GitHubBranch,ParameterValue=${GitHub_Branch} \
  	ParameterKey=GitHubToken,ParameterValue=${GitHub_Token} \
    ParameterKey=PythonBuildVersion,ParameterValue=${Python_Build_Version} \
    ParameterKey=BuildTimeoutMins,ParameterValue=${Build_Timeout_Mins} \
    ParameterKey=InstanceCount,ParameterValue=${Instance_Count} \
    ParameterKey=InstanceType,ParameterValue=${Instance_Type} \
    ParameterKey=MaxRuntimeInSeconds,ParameterValue=${Max_Runtime_In_Seconds} \
    ParameterKey=VolInGB,ParameterValue=${Vol_In_GB} \
  --capabilities CAPABILITY_NAMED_IAM

rm -rf ./LambdaFunction.zip
