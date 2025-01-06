# Out-of-focus-phase-retrieval

## Общая информация

Проект нацелен на создание модели машинного обучения, которая сможет восстанавливать волновой фронт лазерного излучения по распределению интенсивности около фокуса собирающей линзы, при фиксированной амплитуде перед линзой. Затем такая модель может использоваться в паре с деформируемым зеркалом или ЖК-модулятором фазы для управления волновым фронтом и коррекции аберраций излучения.
<p align="center">
  <img src=https://drive.google.com/uc?export=view&id=1k_H6a6TYQBWQoJ5XEueIzTb7ixovNTAx>
</p>

## Обучающая выборка

Обучающая выборка составлялась синтетическим путём, с помощью расчёта распределения амплитуды на некотором расстоянии за линзой в приближении Френеля по формуле [J.W. Goodman. Introduction to Fourier Optics. -M: McGraw-Hill, 1968]:


$A_z(r',\theta')=\hat\nu[\frac{1}{\lambda z}] \hat F (A_0(r,\theta) exp(\frac{ikr^2}{2}(\frac{1}{z}-\frac{1}{f}))$

Где, $\hat F$ оператор взятия Фурье-преобразования, $\hat \nu$ -оператор растяжения, а $A_0$ и $A_z$-комплексные амплитуды перед линзой и на расстоянии $z$ за ней. Волновой фронт мы при этом представляем в виде линейной комбинцаии 18-ти полиномов Цернике, начиная с наклонного астигматизма (Z3). Таким образом, обучающая выборка составлялась для конкетного профился интенсивности перед линзой.
Обучающая выбока составлялась в виде файла y.npy в котором записаны разложения по полиномам Цернике, и папки X, в которую записаны файлы с названиям "n".npy-массив 2x184x184, первая плоскость-модуль комплексной амплитуды за фокусом (первая плоскость) и перед фокусом (вторая плоскость), которые соответствуют n-ой строке в файле y.

## Полученные в ходе работы данные

### Датасет
Обучающая и валидационная выборка [ссылка на архив (5 Гб)](https://drive.google.com/file/d/1GJQNoUuhk4rlNBp1MwmwA5qtDMWiA7mN/view?usp=drive_link) 

Тестовая выбока [ссылка на архив (300 Мб)](https://drive.google.com/file/d/12z94Ac5pduBFlemM_hpoq3cMljnBAW-P/view?usp=sharing) 

### Веса моделей

Использовалась архитектура ConvFormerS18 [arXiv:2210.13452](https://arxiv.org/pdf/2210.13452), у которой последний слой был заменён на линейный с 18-ю выходами. Были обучены модели, которые восстанавливают волновой фронт по распределению интенсивновности снятому:
* [за фокусом](https://drive.google.com/file/d/14dNBQa1X3U5koiVt5LlV7OvEF92hbsHv/view?usp=sharing)
* [перед фокусом](https://drive.google.com/file/d/1GajYZfS6c2na59GuyMGQzx2pgnQGdzs_/view?usp=sharing)
* [с обеих позиций](https://drive.google.com/file/d/1Io8RWLJicvvPHoObCff8O4ACBxBRcQpT/view?usp=sharing)

Все модели весят по 95 Мб

## Описание файлов

### src
#### MathUtils.py
Некоторые математические операции, такие как нормировка разреженных данных, центрирование изображений и полиномы Цернике.
#### FileUtils.py
Функции для работы с изображениями: открытие png, вычитание шума и приведение к необходимому размеру для модели

### Notebooks
#### Sample generator
Для создания обучающей выборки синтетическим путём
#### Train model
Для обучения модели и тестирования модели на синтетических данных.
#### Test model
Для восстановления волнвого фронта полученных в эксперименте пучков.

### data
* после_телескопа.png - профиль интенсивности лазерного излучения перед линзой
* Папка WFS_experiment1: результаты некоторых измерения, проведённые с датчиком Шака-Гартмана. Его показания записаны в папке WFS_data. Остальные файлы-результаты измерений ПЗС-камерой. "+" в названии файла-снято за фокусом, "-" - перед фокусом, "f"-в фокусе.

## Инструкция к использванию

Для восстановления волнового фронта моделью, необходимо скачасть её веса по одной из ссылок выше и поместить в папку weights, находящуюся в одной папке с Notebooks. Затем запустить ноутбук Test_model и, указав путь к необходимому файлу восстновить ВФ.

## Дополнительно

Работа была выполнена при поддержке фонда развития теоретической и математической физики "Базис"
