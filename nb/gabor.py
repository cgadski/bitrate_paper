# %%
import vandc
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2)

df = vandc.fetch().logs
df = df[(df["method"] == "topk") & (df["type"] == "spherical")]
ax[0].matshow(df.pivot(index="k", columns="d", values="acc"))

df = vandc.fetch().logs
df = df[(df["method"] == "topk") & (df["type"] == "gabor")]
ax[1].matshow(df.pivot(index="k", columns="d", values="acc"))

# %%
from project.sparse_recovery import gabor_frame, gabor_frame_v2, rademacher



# %%
df = vandc.fetch().logs
plt.scatter(df["d"], df["energy"], c=df["type"] == "gabor")
