# Runbook: checkout service

Symptoms: elevated HTTP 500 rates on checkout and payment timeouts.

Remediation: first check the payments gateway connection pool; if saturated, scale the checkout deployment and enable the payment retry queue. If errors persist, fail over to the secondary payments provider.

Escalation: page the payments on-call engineer after 15 minutes of elevated errors.
