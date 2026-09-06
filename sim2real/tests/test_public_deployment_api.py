from __future__ import annotations

import sim2real.deploy as deploy
import sim2real.deployment.runner as deployment_runner
import sim2real.validate as validate
import sim2real.deployment.verify as deployment_verify


def test_public_deployment_facade_delegates_to_canonical_runner():
    assert deploy.DeploymentRequest is deployment_runner.SupervisedRequest
    assert deploy.build_deployment_request is deployment_runner._request
    assert deploy.build_deployment_summary is deployment_runner._summary
    assert deploy.execute_deployment is deployment_runner._run_real


def test_public_validation_facade_delegates_to_canonical_verifier():
    assert validate.verify_v94_bundle is deployment_verify.verify_v94_bundle
    assert (
        validate.verify_v94_checkpoint_override
        is deployment_verify.verify_v94_checkpoint_override
    )


def test_public_deployment_dry_run_is_hardware_inert(capsys):
    assert deploy.main(["--policy-rate-hz", "20", "--steps", "2"]) == 0
    output = capsys.readouterr().out
    assert '"hardware_access": false' in output
    assert '"name": "20hz"' in output
    assert '"steps": 2' in output
