import torch
import attention

def test_attention_scores():
    # B=1, N=2, D=4
    a = torch.tensor([[[1.0, 0.0, 1.0, 0.0],  # Token 1 Key
                       [0.0, 2.0, 0.0, 2.0]]]) # Token 2 Key
    b = torch.tensor([[[1.0, 1.0, 1.0, 1.0],  # Token 1 Query
                       [0.0, 1.0, 0.0, 1.0]]]) # Token 2 Query

    # Expected output: (b @ a.T) / sqrt(D), (B, N, N) -> (1, 2, 2)
    expected_output = torch.tensor([[[1.0, 2.0],
                                     [0.0, 2.0]]])

    A = attention.attention_scores(a, b)

    # Note that we use "allclose" and not ==, so we are less sensitive to float inaccuracies.
    assert torch.allclose(A, expected_output)

