# Landmark Identification Approach

### 1. Extract scene attributes and scene categories features using Places365-CNN
```
$ cd places365
$ python run_placesCNN_unified.py image_folder attribute_folder category_folder --attribute-folder-log attribute_logs --category-folder-log category_logs
```

**_Arguments explanation_**:
- **image_folder**: Folder contains scene images 
- **attribute_folder**: Folder contains extracted attribute feature results
- **category_folder**: Folder contains extracted category feature results
- **attribute_logs**: Folder contains human-readable data of attribute feature results (optional)
- **category_logs**: Folder contains human-readable data of category feature results (optional)

