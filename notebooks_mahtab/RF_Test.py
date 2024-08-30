from Cancer_class import Cancer

Cancer.readFile(Cancer, fileName="HAM10000_metadate.csv")
Cancer.getFileHead20(Cancer)
Cancer.dropNaValues(Cancer)
Cancer.dropColumnFromDataFrame(Cancer, 'lesion_id')
Cancer.setTargetVariable(Cancer, 'class')
Cancer.dropColumnFromDataFrame(Cancer, 'class')

Cancer.randomForestModel(Cancer)

