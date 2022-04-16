import matplotlib.pyplot as plt
from rich.table import Table

# 画图函数
def plot_img(images, titles,ing_name, h=62, w=47, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)

    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
    # plt.show()
    plt.savefig("{}.png".format(ing_name))
    # plt.close()

# 设置title
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'Prediction: %s\nReal Name:  %s' % (pred_name, true_name)

# 设置RICH的表格
def make_table(column,row):
    table=Table()
    table.add_column("person name")
    for i in range(len(column)):
        table.add_row(row[i])
    return table
