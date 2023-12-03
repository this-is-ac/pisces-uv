function [UIQM_norm, colrfulness, Value, contrast] =UIQMSharpness(I)
    [UIQM_norm, colrfulness, sharpness, contrast] = UIQM(I);
    Value=sharpness;
end