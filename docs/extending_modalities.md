# Extending modalities

Implement and register the observation-module contract: validate immutable
packets, produce unordered structured measurements, project the current belief,
compute typed innovation/likelihood, and declare exactly which state fields the
measurement supports. Keep sensor feature caches inside the module.

Do not add modality branches to dynamics or replace physical state with sensor
tokens. Add tests for missing/asynchronous packets and same-timestamp ordering.

