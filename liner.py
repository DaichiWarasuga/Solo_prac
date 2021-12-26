#線形回帰
#%%
from sklearn.linear_model import LinearRegression
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.colors
import matplotlib.animation
from IPython.display import HTML
import japanize_matplotlib
#%%
def show_graph(X, Y, x, y):
    fig, ax = plt.subplots(dpi=100)
    ax.scatter(X, Y, marker='.')
    ax.plot(x, y, 'tab:red')
    ax.set_title('線形回帰')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.grid()
    fig.show()
#%%
D = np.array([[1, 3], [3, 6], [6, 5], [8, 7]])
X = D[:,0]
Y = D[:,1]
# %%
reg = LinearRegression()
reg.fit(X.reshape(-1, 1), Y)
# %%
reg.coef_
# %%
reg.intercept_
# %%
reg.score(X.reshape(-1, 1), Y)
# %%
x = np.linspace(0, 35, 100)
y_hat = reg.predict(x.reshape(-1, 1))
show_graph(X, Y, x, y_hat)

# %%
