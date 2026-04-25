from train.grpo_train import parse_action


def test_parse_action_extracts_nested_tool_call_json():
    action = parse_action(
        'prefix {"tool_name":"submit_review","arguments":{"verdict":"reject",'
        '"confidence":0.9,"reasoning":"SQL injection"}} suffix'
    )

    assert action.tool_name == "submit_review"
    assert action.arguments["verdict"] == "reject"
    assert action.arguments["confidence"] == 0.9


def test_parse_action_falls_back_on_malformed_output():
    action = parse_action("not json")

    assert action.tool_name == "submit_review"
    assert action.arguments["verdict"] == "approve"
    assert action.arguments["reasoning"] == "parse error"


def test_parse_action_handles_non_dict_arguments():
    action = parse_action('{"tool_name":"check_security","arguments":"bad"}')

    assert action.tool_name == "check_security"
    assert action.arguments == {}
