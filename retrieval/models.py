from django.db import models


class RetrievalAntiFraud(models.Model):
    id = models.AutoField(primary_key=True)
    feature = models.TextField()
    info = models.TextField()
    status = models.IntegerField()

    class Meta:
        db_table = "retrieval_anti_fraud"
