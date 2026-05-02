import argparse

import boto3
from botocore.exceptions import BotoCoreError, ClientError


def call_bedrock(region: str, model_id: str, prompt: str) -> str:
    client = boto3.client("bedrock-runtime", region_name=region)

    response = client.converse(
        modelId=model_id,
        messages=[
            {
                "role": "user",
                "content": [{"text": "hi"}],
            }
        ],
        inferenceConfig={
            "maxTokens": 120,
            "temperature": 0.2,
        },
    )

    return response["output"]["message"]["content"][0]["text"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Amazon Bedrock locally.")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--model-id", default="amazon.nova-lite-v1:0")
    parser.add_argument(
        "--prompt",
        default="Reply with exactly this text: Bedrock is working.",
    )
    args = parser.parse_args()

    try:
        output = call_bedrock(args.region, args.model_id, args.prompt)
    except ClientError as exc:
        error = exc.response.get("Error", {})
        print(f"Bedrock request failed: {error.get('Code', 'UnknownError')}")
        print(error.get("Message", str(exc)))
        raise SystemExit(1) from exc
    except BotoCoreError as exc:
        print(f"AWS client error: {exc}")
        raise SystemExit(1) from exc

    print("Bedrock response:")
    print(output.strip())


if __name__ == "__main__":
    main()
