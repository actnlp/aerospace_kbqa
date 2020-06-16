import matplotlib.pyplot as plt

from .. import args as config

# 无图形界面需要加，否则plt报错
plt.switch_backend('agg')


def loss_acc_plot(history):
    train_loss = history['train_loss']
    eval_loss = history['eval_loss']
    train_acc = history['train_acc']
    eval_acc = history['eval_acc']

    fig = plt.figure(figsize=(12, 8))
    fig.add_subplot(2, 1, 1)
    plt.title('loss during train')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss)
    plt.plot(epochs, eval_loss)
    plt.legend(['train_loss', 'eval_loss'])

    fig.add_subplot(2, 1, 2)
    plt.title('accuracy during train')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_acc)
    plt.plot(epochs, eval_acc)
    plt.legend(['train_acc', 'eval_acc'])

    plt.savefig(config.plot_path)


def loss_acc_f1_plot(history):
    train_loss = history['train_loss']
    eval_loss = history['eval_loss']
    eval_slot_f1_score = history['eval_slot_f1']
    eval_slot_accuracy = history['eval_slot_acc']
    eval_ent_f1_score = history['eval_ent_f1']
    eval_ent_accuracy = history['eval_ent_acc']
    eval_rel_f1_score = history['eval_rel_f1']
    eval_rel_accuracy = history['eval_rel_acc']
    eval_overall_top1_f1 = history["eval_overall_top1"]
    eval_overall_top3_f1 = history["eval_overall_top3"]

    fig = plt.figure(figsize=(12, 8))
    fig.add_subplot(3, 1, 1)
    plt.title('loss during train and eval')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss)
    plt.plot(epochs, eval_loss)
    plt.legend(['train_loss', 'eval_loss'])

    fig.add_subplot(3, 1, 2)
    plt.title('accuracy during train and eval')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, eval_slot_accuracy)
    plt.plot(epochs, eval_ent_accuracy)
    plt.plot(epochs, eval_rel_accuracy)
    plt.legend(['slot_acc', 'ent_acc', 'rel_acc'])

    fig.add_subplot(3, 1, 3)
    plt.title('f1 during train and eval')
    plt.xlabel('epochs')
    plt.ylabel('f1_score')
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, eval_slot_f1_score)
    plt.plot(epochs, eval_ent_f1_score)
    plt.plot(epochs, eval_rel_f1_score)
    plt.plot(epochs, eval_overall_top1_f1)
    plt.plot(epochs, eval_overall_top3_f1)
    plt.legend(['slot_f1', 'ent_f1', 'rel_f1', 'overall_top1_f1', 'over_all_top3_f1'])

    plt.savefig(config.plot_path)

def one_loss_acc_f1_plot(history):
    train_loss = history['train_loss']
    eval_loss = history['eval_loss']
    eval_ent_f1_score = history['eval_ent_f1']
    eval_ent_accuracy = history['eval_ent_acc']
    eval_rel_f1_score = history['eval_rel_f1']
    eval_rel_accuracy = history['eval_rel_acc']
    eval_overall_top1_f1 = history["eval_overall_top1"]
    eval_overall_top3_f1 = history["eval_overall_top3"]

    fig = plt.figure(figsize=(12, 8))
    fig.add_subplot(3, 1, 1)
    plt.title('loss during train and eval')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss)
    plt.plot(epochs, eval_loss)
    plt.legend(['train_loss', 'eval_loss'])

    fig.add_subplot(3, 1, 2)
    plt.title('accuracy during train and eval')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, eval_ent_accuracy)
    plt.plot(epochs, eval_rel_accuracy)
    plt.legend(['ent_acc', 'rel_acc'])

    fig.add_subplot(3, 1, 3)
    plt.title('f1 during train and eval')
    plt.xlabel('epochs')
    plt.ylabel('f1_score')
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, eval_ent_f1_score)
    plt.plot(epochs, eval_rel_f1_score)
    plt.plot(epochs, eval_overall_top1_f1)
    plt.plot(epochs, eval_overall_top3_f1)
    plt.legend(['ent_f1', 'rel_f1', 'overall_top1_f1', 'over_all_top3_f1'])

    plt.savefig(config.plot_path)


if __name__ == '__main__':
    history = {
        'train_loss': range(100),
        'eval_loss': range(100),
        'train_accuracy': range(100),
        'eval_accuracy': range(100)
    }
    loss_acc_plot(history)
