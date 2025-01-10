from django.db import models

class Story(models.Model):
    title = models.CharField(max_length=200)
    author = models.CharField(max_length=100)
    basic_premise = models.TextField()
    expanded_plot = models.JSONField(null=True,blank=True)  # To store story segments
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.title
