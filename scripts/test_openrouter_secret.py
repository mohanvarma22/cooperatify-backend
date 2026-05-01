import argparse
import json

import boto3
from botocore.exceptions import ClientError


def get_secret(secret_name: str, region_name: str) -> str:
    session = boto3.session.Session()
    client = session.client(
        service_name="secretsmanager",
        region_name=region_name,
    )

    try:
        response = client.get_secret_value(SecretId=secret_name)
    except ClientError:
        raise

    if "SecretString" not in response:
        raise RuntimeError("Secret is binary; expected a string secret.")

    secret = response["SecretString"]

    # If the secret was saved as JSON, support {"OPENROUTER_API_KEY": "..."} too.
    try:
        parsed = json.loads(secret)
    except json.JSONDecodeError:
        return secret

    if isinstance(parsed, dict) and secret_name in parsed:
        return str(parsed[secret_name])

    return secret


def mask_secret(secret: str) -> str:
    if len(secret) <= 8:
        return f"{secret[:2]}..."

    return f"{secret[:4]}...{secret[-4:]}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch OPENROUTER_API_KEY from AWS Secrets Manager."
    )
    parser.add_argument("--secret-name", default="OPENROUTER_API_KEY")
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()

    secret = get_secret(args.secret_name, args.region)
    print(f"{args.secret_name} = {mask_secret(secret)}")


if __name__ == "__main__":
    main()
