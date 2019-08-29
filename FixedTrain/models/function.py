import torch
import torch.nn.functional as F


def batched_kronecker(matrix1, matrix2):
    matrix1_flatten = matrix1.reshape(matrix1.size()[0], -1)
    matrix2_flatten = matrix2.reshape(matrix2.size()[0], -1)
    return torch.bmm(matrix1_flatten.unsqueeze(2), matrix2_flatten.unsqueeze(1)).reshape(
        [matrix1.size()[0]] + list(matrix1.size()[1:]) + list(matrix2.size()[1:])).permute(
        [0, 1, 3, 2, 4]).reshape(matrix1.size(0), matrix1.size(1) * matrix2.size(1),
                                 matrix1.size(2) * matrix2.size(2))



def computeGramMatrix(A, B):
    """
    Constructs a linear kernel matrix between A and B.
    We assume that each row in A and B represents a d-dimensional feature vector.

    Parameters:
      A:  a (n_batch, n, d) Tensor.
      B:  a (n_batch, m, d) Tensor.
    Returns: a (n_batch, n, m) Tensor.
    """

    assert (A.dim() == 3)
    assert (B.dim() == 3)

    assert (A.size(0) == B.size(0) and A.size(2) == B.size(2))

    return torch.bmm(A, B.transpose(1, 2))


def binv(b_mat):
    """
    Computes an inverse of each matrix in the batch.
    Pytorch 0.4.1 does not support batched matrix inverse.
    Hence, we are solving AX=I.

    Parameters:
      b_mat:  a (n_batch, n, n) Tensor.
    Returns: a (n_batch, n, n) Tensor.
    """

    id_matrix = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
    id_matrix = id_matrix.cuda()
    b_inv, _ = torch.gesv(id_matrix, b_mat)

    return b_inv

def relation_loss(embtypes, emb_support):
    n_class = embtypes.shape[1]

    loss = []
    for i in range(n_class):
        embclass = embtypes[:,i,:].unsqueeze(1)
        dis = torch.pow(embclass-embtypes, 2).sum(dim=-1)
        # dis = dis / 16000
        dis = dis / 1024
        dis = -(dis.sum(dim=-1)/(2 * 4))
        entropy = torch.exp(dis).unsqueeze(1)
        loss.append(entropy)

    loss = torch.cat(loss, dim=-1)

    return loss.mean()


# def relation_loss(prototypes):
#     '''
#         prototypes ： 【2，5，16000】
#         query ： 【2，30， 16000】
#         AB ： 【2， 30， 5】
#         AA ： 【2， 30 ，1】
#         BB ： 【2， 1 ，5】
#         logits first： 【2， 30， 5】
#     '''
#
#     proto_loss = []
#     for bs in range(prototypes.shape[0]):
#         loss = []
#         for i in range(prototypes.shape[1]):
#             proto_a = prototypes[bs][i]
#             for j in range(i+1, prototypes.shape[1]):
#                 proto_b = prototypes[bs][j]
#                 dis = -((proto_a-proto_b) ** 2).sum()
#                 entropy = torch.exp(dis / 16000)
#                 entropy = entropy.unsqueeze(0)
#                 loss.append(entropy)
#         loss = torch.cat(loss, dim=0)
#         # print('batch proto loss', loss)
#
#         proto_loss.append(loss.mean().unsqueeze(0))
#     proto_loss = torch.cat(proto_loss).mean()
#     # print('proto loss', proto_loss)
#
#     # loss = gen_loss + proto_loss
#     loss = proto_loss
#     return loss



def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.

    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth]))
    encoded_indicies = encoded_indicies.cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)

    return encoded_indicies


if __name__ == '__main__':
    pass