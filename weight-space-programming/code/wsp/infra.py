"""Provision / tear down the worker.

  python wsp/infra.py up       # bucket, uploads, IAM role, SG, g5.12xlarge, CloudWatch stop alarm
  python wsp/infra.py status
  python wsp/infra.py down     # terminate instance, delete alarm (bucket kept: it holds results)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wsp.common import BUCKET, INSTANCE_NAME, MODEL_REV, PRACTICE_DIR, PROJECT_TAG, REGION, ROOT

INSTANCE_TYPE = "g5.xlarge"
ROLE = "weightspace-worker-role"
SG = "weightspace-worker-sg"
ALARM = "weightspace-worker-idle-stop"
TAGS = [{"Key": "Name", "Value": INSTANCE_NAME}, {"Key": "Project", "Value": PROJECT_TAG}]
sess = boto3.Session(region_name=REGION)
ec2, s3, iam, ssm, cw = (sess.client(x) for x in ("ec2", "s3", "iam", "ssm", "cloudwatch"))


def log(m):
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


def bucket() -> None:
    try:
        s3.head_bucket(Bucket=BUCKET)
        log(f"bucket {BUCKET} exists")
    except s3.exceptions.ClientError:
        s3.create_bucket(Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": REGION})
        s3.put_public_access_block(Bucket=BUCKET, PublicAccessBlockConfiguration=dict(
            BlockPublicAcls=True, IgnorePublicAcls=True, BlockPublicPolicy=True, RestrictPublicBuckets=True))
        s3.put_bucket_tagging(Bucket=BUCKET, Tagging={"TagSet": [{"Key": "Project", "Value": PROJECT_TAG}]})
        log(f"bucket {BUCKET} created")
    for p in [ROOT / "remote" / "worker.py", ROOT / "remote" / "prepare_heldout.py", ROOT / "wsp" / "common.py"]:
        s3.upload_file(str(p), BUCKET, f"code/{p.name}")
    for p in PRACTICE_DIR.glob("*.jsonl"):
        s3.upload_file(str(p), BUCKET, f"practice/{p.name}")
    log("code + practice uploaded")


def role() -> str:
    trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"},
                                                     "Action": "sts:AssumeRole"}]}
    try:
        iam.get_role(RoleName=ROLE)
        log(f"role {ROLE} exists")
    except iam.exceptions.NoSuchEntityException:
        iam.create_role(RoleName=ROLE, AssumeRolePolicyDocument=json.dumps(trust),
                        Tags=[{"Key": "Project", "Value": PROJECT_TAG}])
        iam.attach_role_policy(RoleName=ROLE, PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore")
        iam.put_role_policy(RoleName=ROLE, PolicyName="wsp-bucket", PolicyDocument=json.dumps({
            "Version": "2012-10-17", "Statement": [
                {"Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": f"arn:aws:s3:::{BUCKET}"},
                {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject"], "Resource": f"arn:aws:s3:::{BUCKET}/*"}]}))
        iam.create_instance_profile(InstanceProfileName=ROLE)
        iam.add_role_to_instance_profile(InstanceProfileName=ROLE, RoleName=ROLE)
        log(f"role + instance profile {ROLE} created; waiting for IAM propagation")
        time.sleep(15)
    return ROLE


def security_group() -> str:
    vpc = ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])["Vpcs"][0]["VpcId"]
    r = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [SG]}, {"Name": "vpc-id", "Values": [vpc]}])
    if r["SecurityGroups"]:
        return r["SecurityGroups"][0]["GroupId"]
    g = ec2.create_security_group(GroupName=SG, Description="weightspace worker: no inbound, SSM only", VpcId=vpc,
                                  TagSpecifications=[{"ResourceType": "security-group", "Tags": TAGS}])
    log(f"security group {g['GroupId']} created (no inbound rules)")
    return g["GroupId"]


def find() -> list[dict]:
    r = ec2.describe_instances(Filters=[{"Name": "tag:Name", "Values": [INSTANCE_NAME]},
                                        {"Name": "instance-state-name",
                                         "Values": ["pending", "running", "stopping", "stopped"]}])
    return [i for res in r["Reservations"] for i in res["Instances"]]


def up() -> None:
    ident = sess.client("sts").get_caller_identity()
    log(f"account {ident['Account']} as {ident['Arn']}")
    if find():
        sys.exit(f"{INSTANCE_NAME} already exists: {[i['InstanceId'] for i in find()]}")
    bucket()
    prof = role()
    sg = security_group()
    ami = ssm.get_parameter(Name="/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id")["Parameter"]["Value"]
    img = ec2.describe_images(ImageIds=[ami])["Images"][0]
    log(f"AMI {ami}: {img['Name']}")
    ud = (ROOT / "remote" / "setup.sh").read_text().replace("__BUCKET__", BUCKET).replace(
        "__REGION__", REGION).replace("__MODEL_REV__", MODEL_REV)
    vpc = ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])["Vpcs"][0]["VpcId"]
    offered = {o["Location"] for o in ec2.describe_instance_type_offerings(
        LocationType="availability-zone", Filters=[{"Name": "instance-type", "Values": [INSTANCE_TYPE]}])["InstanceTypeOfferings"]}
    subnets = [s for s in ec2.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc]},
                                                        {"Name": "default-for-az", "Values": ["true"]}])["Subnets"]
               if s["AvailabilityZone"] in offered]
    tries = [(sn, a) for sn in subnets for a in range(2)]
    for attempt, (sn, _) in enumerate(tries):
        try:
            log(f"trying {INSTANCE_TYPE} in {sn['AvailabilityZone']}")
            r = ec2.run_instances(SubnetId=sn["SubnetId"],
                ImageId=ami, InstanceType=INSTANCE_TYPE, MinCount=1, MaxCount=1, UserData=ud,
                IamInstanceProfile={"Name": prof}, SecurityGroupIds=[sg],
                InstanceInitiatedShutdownBehavior="stop",
                MetadataOptions={"HttpTokens": "required", "HttpEndpoint": "enabled"},
                BlockDeviceMappings=[{"DeviceName": img["RootDeviceName"],
                                      "Ebs": {"VolumeSize": 200, "VolumeType": "gp3", "DeleteOnTermination": True}}],
                TagSpecifications=[{"ResourceType": "instance", "Tags": TAGS}, {"ResourceType": "volume", "Tags": TAGS}])
            break
        except ec2.exceptions.ClientError as e:
            if attempt < len(tries) - 1 and ("Invalid IAM Instance Profile" in str(e)
                                             or "InsufficientInstanceCapacity" in str(e)):
                log(f"{str(e)[:160]} -> retrying")
                time.sleep(10)
                continue
            raise
    iid = r["Instances"][0]["InstanceId"]
    log(f"launched {iid} ({INSTANCE_TYPE}) in {r['Instances'][0]['Placement']['AvailabilityZone']}")
    cw.put_metric_alarm(
        AlarmName=ALARM, Namespace="AWS/EC2", MetricName="CPUUtilization",
        Dimensions=[{"Name": "InstanceId", "Value": iid}], Statistic="Average", Period=600,
        EvaluationPeriods=3, Threshold=1.0, ComparisonOperator="LessThanThreshold",
        AlarmActions=[f"arn:aws:automate:{REGION}:ec2:stop"], TreatMissingData="notBreaching",
        Tags=[{"Key": "Project", "Value": PROJECT_TAG}])
    log(f"CloudWatch alarm {ALARM}: stop if CPU < 1% for 30 min")


def status() -> None:
    for i in find():
        log(f"{i['InstanceId']} {i['InstanceType']} {i['State']['Name']} launched {i['LaunchTime']}")
    for k in ("status/setup_done",):
        try:
            log(f"{k}: {s3.get_object(Bucket=BUCKET, Key=k)['Body'].read().decode().strip()}")
        except Exception:
            log(f"{k}: not yet")


def down() -> None:
    ids = [i["InstanceId"] for i in find()]
    if ids:
        ec2.terminate_instances(InstanceIds=ids)
        ec2.get_waiter("instance_terminated").wait(InstanceIds=ids)
        log(f"terminated {ids}")
    cw.delete_alarms(AlarmNames=[ALARM])
    left = find()
    log(f"remaining {INSTANCE_NAME} instances: {len(left)}")


if __name__ == "__main__":
    {"up": up, "status": status, "down": down}[sys.argv[1]]()
