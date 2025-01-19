import torch
import gpytorch
import matplotlib.pyplot as plt
import numpy as np

# sparse gaussian
class SparseGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(SparseGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# 初始化数据
np.random.seed(42)
torch.manual_seed(42)

# 初始训练数据
X_train = torch.tensor([[0.0]], dtype=torch.float32)
Y_train = torch.sin(X_train) + torch.normal(0, 0.1, X_train.size())

# 定义诱导点（可以选择初始的一部分数据或预先定义）
inducing_points = torch.linspace(0, 10, 20).unsqueeze(-1)

# 实例化模型和导数器
model = SparseGPModel(inducing_points)
likelihood = gpytorch.likelihoods.GaussianLikelihood()

# 训练模型函数
def train_model(model, likelihood, X_train, Y_train, training_iter=50):
    model.train()
    likelihood.train()

    # 使用Adam优化器
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=0.01)

    # 使用变分ELBO作为损失函数
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=Y_train.size(0))

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, Y_train)
        loss.backward()
        optimizer.step()
        # if (i+1) % 10 == 0:
        #     print(f'Iter {i+1}/{training_iter} - Loss: {loss.item():.3f}')

# 初始训练
train_model(model, likelihood, X_train, Y_train)

# 在线更新函数
def online_update(model, likelihood, new_x, new_y, X_train, Y_train, training_iter=10):
    # 添加新数据
    X_train = torch.cat([X_train, new_x], dim=0)
    Y_train = torch.cat([Y_train, new_y], dim=0)
    # 重新训练模型
    train_model(model, likelihood, X_train, Y_train, training_iter)
    return X_train, Y_train

# 生成测试数据
X_test = torch.linspace(0, 10, 1000).unsqueeze(-1)
model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    y_pred = likelihood(model(X_test)).mean

# 绘制初始预测
plt.figure(figsize=(10,6))
plt.plot(X_test.numpy(), y_pred.numpy(), label='Initial Prediction')
plt.scatter(X_train.numpy(), Y_train.numpy(), color='red', label='Training Data')
plt.legend()
plt.show()

# 模拟逐步获取新数据
for i in range(10):
    # 获取新的数据点（例如，按顺序获取）
    new_x = torch.tensor([[i + 1.0]], dtype=torch.float32)
    new_y = torch.sin(new_x) + torch.normal(0, 0.1, new_x.size())
    X_train, Y_train = online_update(model, likelihood, new_x, new_y, X_train, Y_train, training_iter=20)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        y_pred = likelihood(model(X_test)).mean
    # 绘制更新后的预测
    plt.figure(figsize=(10,6))
    plt.plot(X_test.numpy(), y_pred.numpy(), label='Updated Prediction')
    plt.scatter(X_train.numpy(), Y_train.numpy(), color='red', label='Training Data')
    plt.legend()
    plt.title(f'After adding data point {i+1}')
    plt.show()
