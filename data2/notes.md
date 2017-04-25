ad.data - сделать
ad1000.data - сделать
cancer-wisconsin.data - очень плохое начальное приближение.
car0.data - задача лучше решается байесом чем SVM на 10%
car1.data - +3 % от начального приближения. SVM лучше еще на ~4%
car2.data - байес работает лучше чем svm на 3  еще на 3 улучшилось
heart_scale.data - сравнимо
iris0.data - сравнимо
iris1.data - сравнимо
iris2.data - сравнимо
liver.data - c функцией потерь Q +7 лучше и быстрее svm +30 к нач приближению
mushrooms2000.data - полохое нач приближение
sonar.data - + 10 - 15  но хуже svm
spambase.data
uni_grman.data
vlada.data
vlada2.data
wdbc.data
wine0.data
wine1.data
wine2.data
wpbc.data

# не сдвигается с начального приближения. SVM всегда стабильно лучше
segmentaion-BRICKFACE.data
segmentaion-CEMENT.data
segmentaion-FOLIAGE.data
segmentaion-GRASS.data
segmentaion-PATH.data
segmentaion-SKY.data
segmentaion-WINDOW.data


-segmentaion-BRICKFACE/CEMENT/FOLIAGE/GRASS/PATH/SKY/WINDOW: классификация фотокграфий, один против всех 
http://archive.ics.uci.edu/ml/datasets/Image+Segmentation


ad500: интернет реклама, случайные 500 объектов
http://archive.ics.uci.edu/ml/datasets/Internet+Advertisements

car0/car1/car2: оценка машин
Эксперимены по признаку
unacc vs acc
unacc vs good
unacc vs vgood
http://archive.ics.uci.edu/ml/datasets/Car+Evaluation

heart_scale: тренировачный датасет от одной реализации SVM
https://github.com/cjlin1/libsvm/blob/master/heart_scale

iris0/iris1/iris2: Ирисы, все попарные сравнения классов
http://archive.ics.uci.edu/ml/datasets/Iris

liver: Растройства печени, здесь получился большой выигрыш при использовании X2
http://archive.ics.uci.edu/ml/datasets/Liver+Disorders

mushrooms2000: съедобные/несъедобные грибы случайные 2000 объектов
http://archive.ics.uci.edu/ml/datasets/Mushroom

sonar: по сигналам отличать маталические объекты от каменных
http://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks)

spambase1000: web спам
http://archive.ics.uci.edu/ml/datasets/Spambase

-uni_grman: кредитные истории в германии
http://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data)

wine0/wine1/wine2: происхождение вина, каждый с каждым
http://archive.ics.uci.edu/ml/datasets/Wine

cancer-wisconsin/wdbc/wpbc: 3 датасета про рак груди
http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29