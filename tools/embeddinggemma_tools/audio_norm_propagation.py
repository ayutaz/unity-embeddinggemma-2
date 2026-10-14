"""Independent diagnostic error budget; never changes the pinned reference."""


def rms_norm_oracle(values, folded_weight, epsilon, *, restore_multiplier=2):
    raise NotImplementedError("independent RMSNorm oracle not implemented")


def norm_error_budget(propagated_input, propagated_output, native_input, native_output,
                      folded_weight, epsilon, *, restore_multiplier=2):
    raise NotImplementedError("propagated-input error budget not implemented")
