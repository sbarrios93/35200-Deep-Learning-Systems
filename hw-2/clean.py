import pandas as pd
import results as results


data = results.results1 | results.results2 | results.results3 | results.results4
df = pd.DataFrame(data)

df.index = ['train_time',  'test_time', 'acurracy', 'history']
df = df.T

df2 = pd.DataFrame(columns=['optimizer', 'activation', 'normalization', 'schedule'])
for i in df.index:
    x = i.split('_')
    x_post = x[-1]
    x = x[0:3]
    x.append(x_post)
    df2 = pd.concat([df2, pd.DataFrame(x, index=['optimizer', 'activation', 'normalization', 'schedule']).T])

df2.reset_index(inplace=True, drop=True)
df = df2.join(df.reset_index(drop=True), how='outer')


def activation(x):
    if "<lambda>" in x.split():
        return 'lrelu'
    else:
        return x
def schedule(x):
    if "schedule.CosineDecay" in x.split():
        return "Cosine Decay"
    elif "schedule.ExponentialDecay" in x.split():
        return "Exponential Decay"
    else:
        return x
def normalization(x):
    if x == "[0 0 0 0]":
        return "No Normalization"
    else:
        return "Full Normalization"
df.activation = df.apply(lambda x: activation(x.activation), axis=1)
df.schedule = df.apply(lambda x: schedule(x.schedule), axis=1)
df.normalization = df.apply(lambda x: normalization(x.normalization), axis=1)
df.history = df.apply(lambda x: len(x.history["accuracy"]), axis=1)
df["time_per_epoch"] = df.train_time/df.history
df.sort_values(by=['acurracy'], inplace=True)
df