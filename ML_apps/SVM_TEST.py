
from river import anomaly
from river import compose
from river import datasets
from river import metrics
from river import preprocessing
X_y = [
     ({"cpu_usage": 0.9, "ram_usage": 0.3, "n_process": 200, "elasticsearch_cpu_usage": 0.5,
         "incoming_connexions_3_minutes": 10}, 0),
     ({"cpu_usage": 0.9, "ram_usage": 0.3, "n_process": 200, "elasticsearch_cpu_usage": 0.5,
         "incoming_connexions_3_minutes": 10}, 0),
     ({"cpu_usage": 0.99, "ram_usage": 0.99, "n_process": 200, "elasticsearch_cpu_usage": 0.99,
         "incoming_connexions_3_minutes": 200}, 1),  # OUTLIER
     ({"cpu_usage": 0.9, "ram_usage": 0.3, "n_process": 200, "elasticsearch_cpu_usage": 0.0,
         "incoming_connexions_3_minutes": 10}, 0),
     ({"cpu_usage": 0.9, "ram_usage": 0.3, "n_process": 900, "elasticsearch_cpu_usage": 0.5,
         "incoming_connexions_3_minutes": 10}, 0),
    ]

model = anomaly.QuantileThresholder(
    anomaly.OneClassSVM(nu=0.2),
    q = 0.995
     )

auc = metrics.ROCAUC()
print (auc)
for x, y in X_y:
     score = model.score_one(x)

     model = model.learn_one(x)
     auc = auc.update(y, score)

print (auc)


