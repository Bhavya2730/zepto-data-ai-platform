import json
import urllib.request


API_URL = "http://127.0.0.1:8000/ask"


def ask(query: str):
    payload = json.dumps({"query": query}).encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    policy_query = "What is the delivery fee?"
    general_query = "What is the capital of France?"

    print("\nPolicy query:")
    print(policy_query)

    policy_response = ask(policy_query)

    print(json.dumps(policy_response, indent=2))

    print("\nGeneral query:")
    print(general_query)

    general_response = ask(general_query)

    print(json.dumps(general_response, indent=2))

    if not policy_response["answer"]:
        raise AssertionError("Policy response has no answer.")

    if not policy_response["sources"]:
        raise AssertionError("Policy response has no sources.")

    if general_response["sources"]:
        raise AssertionError(
            "General response should have no sources."
        )

    print("\nStage 3.7 API verification passed.")


if __name__ == "__main__":
    main()