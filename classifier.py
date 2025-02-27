import tensorflow as tf
from xgboost import XGBClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

def mlp(X_train, y_train, epoch=100, n_hidden=500, n_classes=6, dropout=0.5):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(n_hidden, activation='relu'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(n_hidden, activation='relu'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epoch, verbose=0)
    # predict_softmax = model.predict(X_test)  
    # predict_label = encoder.inverse_transform(predict_softmax.argmax(axis=1))
    # score = accuracy_score(y_test, predict_label)
    return model

def cnn(X_train, y_train, epoch=100, n_hidden=500, n_classes=6, dropout=0.5):
    model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(5, 1)),
    tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Dropout(dropout),
    tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=1),  # Adjusted pool size
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(n_hidden, activation='relu'),
    tf.keras.layers.Dropout(dropout),
    tf.keras.layers.Dense(n_hidden, activation='relu'),
    tf.keras.layers.Dropout(dropout),
    tf.keras.layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epoch, verbose=0)
    # predict_softmax = model.predict(X_test)  
    # predict_label = encoder.inverse_transform(predict_softmax.argmax(axis=1))
    # score = accuracy_score(y_test, predict_label)
    return model

def xgboost(X_train, y_train):
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    # score = accuracy_score(y_test, y_pred)
    return model

def decision_tree(X_train, y_train):
    model = tree.DecisionTreeClassifier(max_depth=3)
    model.fit(X_train, y_train)
    return model

def random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    return model

def adaboost(X_train, y_train):
    model = AdaBoostClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def svm(X_train, y_train, kernel='linear', gamma=0.1, C=20):
    model = SVC(kernel=kernel, gamma=gamma, C=C)
    model.fit(X_train, y_train)
    return model

def naive_bayes(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train) 
    return model
