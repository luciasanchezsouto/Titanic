import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import scipy.stats as stats
import plotly.graph_objects as go
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="El desastre del Titanic", page_icon="🚢",layout="wide")
@st.cache_resource #decorador para cachear los datos
def cargar_datos(): #función para cargar los datos
    return pd.read_csv("https://github.com/luciasanchezsouto/Titanic/blob/49132077710370ec35ab941c409fb9c3947a9863/titanic.csv")

# Barra lateral con opciones
st.sidebar.title('Menú')
page = st.sidebar.selectbox('Selecciona una página:', ['Inicio', 'Sobre el Titanic', 'Descripción de la muestra', 'Procedimiento de análisis', 'Análisis descriptivo', 'Análisis causal', 'Conclusiones'])


df = pd.read_csv(https://github.com/luciasanchezsouto/Titanic/blob/49132077710370ec35ab941c409fb9c3947a9863/titanic.csv)

# Ejecutar código basado en la página seleccionada
if page == 'Inicio':
    st.title("Inicio")
    # Aquí puedes agregar el código que quieres que se ejecute en la página de inicio
    st.write("¿Qué pasó con el Titanic?")
    st.write("Una exploración a través de los datos")
    # Calcula el ancho de las columnas laterales para centrar la imagen principal
    col1, col2, col3 = st.columns([1,6,1])
    with col2: # Usa esta columna para mostrar la imagen centrada
        st.image("https://maritimecyprus.com/wp-content/uploads/2022/04/Titanic-in-port.jpg", use_column_width=True)  
    

elif page == 'Sobre el Titanic':
    st.title("Sobre el Titanic")
    st.write("El Titanic fue el mayor barco transatlántico en cuanto a capacidad de pasajeros en el momento de finalización de su construcción (3000 personas). En su viaje inaugural de Southampton a Nueva York, 2223 personas que viajaban a bordo vieron cómo este, en la noche del 14 de abril de 1912, se hundía tras un choque con un iceberg y morían 1514 de estas personas.")
    st.write("A pesar de la gran capacidad del barco, solo contaba con 20 botes salvavidas, siendo la capacidad total 1178 (muy por debajo de los 2223 que viajaban y por supuesto por debajo de las 3000 plazas ofertadas). Tampoco había suficientes chalecos salvavidas, siendo muchos de mala calidad. Esto provocó la fatídica muerte de más de un 75% de los viajeros por ahogamiento o hipotermia.")
    # Calcula el ancho de las columnas laterales para centrar la imagen principal
    col1, col2, col3 = st.columns([1,6,1])
    with col2: # Usa esta columna para mostrar la imagen centrada
        st.image("https://www.thoughtco.com/thmb/0DGYC6oO3twSpYSqG1Fk8gzJtFY=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/GettyImages-517357578-5c4a27edc9e77c0001ccf77d.jpg", use_column_width=True)
    

elif page == 'Descripción de la muestra':
    st.title("Descripción de la muestra")
    st.write("Dimensiones de la muestra")
    st.write("891 filas o pasajeros")
    st.write("12 columnas o variables: 'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age' 'Number of Siblings/Spouses Aboard', 'Number of Parents/Children Aboard', 'Ticket', 'Fare', 'Cabin', 'Embarked'")
    st.dataframe (df.head(10))
    st.write ("Número de filas y columnas: ", df.shape)
    
    
elif page == 'Procedimiento de análisis':
    st.title("Procedimiento de análisis")
    st.write("1. Carga de datos del archivo CSV")
    st.write("2. Limpieza de datos")
    st.write("3. Análisis descriptivo")
    st.write("4. Análisis causal")
    
    st.write("--> Valores nulos en porcentaje en el dataset:")
    st.dataframe(df.isnull().sum() / len(df) * 100)
    
    #Valores nulos porcentaje
    valores_nulos_porcentaje = df.isnull().sum() / len(df) * 100
    fig = px.bar(x=valores_nulos_porcentaje.index, y=valores_nulos_porcentaje.values, labels={'x': 'Columnas', 'y': 'Porcentaje sobre categoría'}, title='Porcentaje de valores nulos sobre 100%')
    fig.update_layout(title_x=0.5)
    fig.update_yaxes(range=[0, 100])
    st.plotly_chart(fig)

    #Distribución de datos de la columna "Age"
    plt.figure(figsize=(16, 8))
    sns.histplot(df['Age'])
    plt.title('Distribución de la columna "Age"', size=25)
    plt.ylabel('Conteo');
    st.pyplot(plt)
    
    from scipy.stats import shapiro
    stat, p = shapiro(df['Age'].dropna()) # Usamos dropna() para eliminar valores NaN
    print('Estadístico de prueba (W):', stat)
    print('Valor p:', p)
    st.write("Según el test de Shapiro, y dado el p-valor<0.05 (7.337348958673592e-08), hay evidencia suficiente para rechazar la hipótesis nula de normalidad. Por su forma, recuerda a una F de Snedecor.")
    st.write("Sustituimos los NaN por la mediana de la columna 'Age'.")
    mediana_edad = df['Age'].median()
    st.write(f"La mediana de la edad es: {mediana_edad}.")
    
    #Distribución embarked
    image_path = r'C:\Users\lucia\Desktop\UPGRADE_works\Entrega_1\graf_embarked.png'
    st.image(image_path)
    st.write("Sustituimos los NaN por 'Sin datos de embarque'.")
    
    st.write("Ahora ya no tenemos valores nulos en las columnas 'Age' y 'Embarked'.")
    
    #Distribución de quién tiene cabina por clase de pasajeros
    st.write("-->Distribución de quién tiene cabina por clase de pasajeros:")
    df['Has_Cabin'] = df['Cabin'].apply(lambda x: 'Yes' if pd.notna(x) else 'No')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='Pclass', hue='Has_Cabin', data=df, ax=ax)
    ax.set_title('Presencia de Información de Cabina por Clase de Pasajero')
    ax.set_xlabel('Clase de Pasajero')
    ax.set_ylabel('Conteo de Pasajeros')
    ax.legend(title='Información de Cabina', loc='upper right')
    st.pyplot(fig)
    st.write("Los pasajeros de primera clase tenían asignada una cabina con todo tipo de lujos, mientras que los de segunda y tercera clase compartían camarotes.")
    
    
elif page == 'Análisis descriptivo':
    st.title("Análisis descriptivo")
    st.write("Sobre todo usando las columnas 'Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked'")
    st.write("Visualización de datos: gráficos")
    
    #Edad mínima y máxima de las personas del barco
    st.write("-->Edad mínima y máxima de las personas del barco:")
    edad_máxima = df['Age'].max()
    edad_mínima = df['Age'].min()
    st.write(f"Edad mínima: {edad_mínima}, Edad máxima: {edad_máxima}")
    
    #Precio más bajo y más alto de los tickets
    st.write("-->Precio más bajo y más alto de los tickets:")
    precio_mínimo = df['Fare'].min()
    precio_máximo = df['Fare'].max()
    st.write(f"Precio mínimo: {precio_mínimo}, Precio máximo: {precio_máximo}")
    #Libras de 1912 a 2024
    st.write("-->Pasamos los precios en libras de 1912 a libras de 2024")
    Factor_conversión = 95.42 # 1 libra de 1912 equivale a 95.42 libras de 2024, según https://www.bankofengland.co.uk/monetary-policy/inflation/inflation-calculator
    df['Fare'] = df['Fare'] * Factor_conversión
    st.write("Tarifas actualizadas (ajustadas a 2024):")
    st.dataframe(df['Fare'])
    #Precio mínimo y máximo de la columna "Fare"
    precio_mínimo = df['Fare'].min()
    precio_máximo = df['Fare'].max()
    st.write(f"Precio mínimo: {precio_mínimo}, Precio máximo: {precio_máximo}")
    
    #Embarcados por género
    st.write("-->Embarcados por género:")
    x_values = ['Mujer', 'Hombre']
    y_values = [314, 577]
    fig = go.Figure(data=[go.Bar(x=x_values, y=y_values, text=y_values, textposition='auto')])
    fig.update_layout(title='Embarcados por género', title_x=0.5)
    fig.update_yaxes(range=[0, 600])
    st.plotly_chart(fig)
    
    #Supervivientes por tramo de edad
    st.write("-->Supervivientes por tramo de edad:")
    st.write("Tramos de edad: Niño (0-12 años), Adolescente (13-18 años), Joven (18-27 años), Adulto (28-60 años), Persona mayor (60 años en adelante)") #según https://www.minsalud.gov.co/proteccionsocial/Paginas/cicloVida.aspx#:~:text=La%20siguiente%20clasificaci%C3%B3n%20es%20un,(60%20a%C3%B1os%20y%20m%C3%A1s).
    bins = [0, 12, 18, 27, 60, 80]
    labels = ['Niño', 'Adolescente', 'Joven', 'Adulto', 'Persona mayor']
    df['Edad_Tramo'] = pd.cut(df['Age'], bins=bins, labels=labels) 
    supervivientes_por_tramo_edad_porcentual = df[df['Survived'] == 1]['Edad_Tramo'].value_counts() / df['Edad_Tramo'].value_counts() * 100
    fig = px.bar(x=df['Edad_Tramo'].value_counts().index, y=supervivientes_por_tramo_edad_porcentual.values, labels={'x': 'Tramo de Edad', 'y': 'Porcentaje de Supervivientes'}, title='Porcentaje de supervivientes por tramo de edad sobre 100%')
    fig.update_layout(title_x=0.5)
    fig.update_yaxes(range=[0, 100])
    st.plotly_chart(fig)
    
    #Supervivientes por género
    st.write("-->Supervivientes por género:")
    x_values = ['Mujer', 'Hombre']
    y_values = [74.203822, 18.890815]
    fig = go.Figure(data=[go.Bar(x=x_values, y=y_values, text=y_values, textposition='auto')])
    fig.update_layout(title='Porcentaje de supervivencia por género', title_x=0.5)
    fig.update_yaxes(range=[0, 100])
    st.plotly_chart(fig)
    
    #Supervivientes por clase
    st.write("-->Supervivientes por clase:")
    x_values = ['Clase 1', 'Clase 2', 'Clase 3']
    y_values = [62.962963, 47.282609, 24.236253]
    fig = go.Figure(data=[go.Bar(x=x_values, y=y_values, text=y_values, textposition='auto')])
    fig.update_layout(title='Porcentaje de supervivencia por clase', title_x=0.5)
    fig.update_yaxes(range=[0, 100])
    st.plotly_chart(fig)
    
    #Porcentaje de Pasajeros de Primera Clase por Género en %
    st.write("-->Porcentaje de Pasajeros de Primera Clase por Género en %:")
    total_gender = df['Sex'].value_counts()
    first_class_gender = df[df['Pclass'] == 1]['Sex'].value_counts()
    percentage_first_class = (first_class_gender / total_gender) * 100
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=percentage_first_class.index, y=percentage_first_class.values, ax=ax)
    ax.set_title('Porcentaje de Pasajeros de Primera Clase por Género en %')
    ax.set_xlabel('Género')
    ax.set_ylabel('Porcentaje de Primera Clase (%)')
    st.pyplot(fig)
    
    #Correlación entre vaiables
    st.write("-->Correlación entre variables:")
    def main():
        st.title('Heatmap de Correlación')

    # Calcula la matriz de correlación
        corr_matrix = df.corr()
    # Crea el gráfico
        fig, ax = plt.subplots(figsize=(10, 8))  # Ajusta el tamaño según necesites
        sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, ax=ax)
    # Muestra el gráfico en Streamlit
        st.pyplot(fig)
    if __name__ == "__main__":
        main()
    
    
    
elif page == 'Análisis causal':
    st.title("Análisis causal")
    st.write("¿Qué variables influyeron en la supervivencia de los pasajeros?: Simple Difference in Outcomes (SDO) o Average Treatment Effect (ATE), Subclassification")
    st.write("Sabemos que sobrevivir al naufragio del Titanic estuvo relacionado con la clase de pasajero, el género y la edad.")
    st.write ("No obstante, si queremos saber cuál es la ventaja de supervivencia de los que iban en primera clase y al mismo tiempo los niños y mujeres tenían prioridad en ir en esta clase, puede ser que la respuesta que bvuscamos simplemente capture este efecto.")
    st.write ("Primero, calcularemos una diferencia simple en los resultados (SDO), es decir, un Average Treatment Effect (ATE).")
    st.write ("E[Y|D=1] - E[Y|D=0]")
    st.write ("donde Y es la variable de supervivencia y D es la variable de tratamiento (ser de primera clase).")
    df['FirstClass'] = df['Pclass'].apply(lambda x: 1 if x == 1 else 0)
    # Calcular la media de supervivencia en primera clase y no primera clase
    mean_survival_first_class = 100 * df[df['FirstClass'] == 1]['Survived'].mean()
    mean_survival_non_first_class = 100 * df[df['FirstClass'] == 0]['Survived'].mean()
    st.write(f'Mean Survival in First Class: {mean_survival_first_class:.2f}%')
    st.write(f'Mean Survival in Non-First Class: {mean_survival_non_first_class:.2f}%')
    st.write("La diferencia simple en los resultados (SDO) es de 40.13%.")
    #Ponderación de subclasificación
    st.write ("Sin embargo, dado que este resultado no tiene en cuenta los factores observables de edad y sexo, se trata de una estimación sesgada de la ATE. Por tanto, a continuación utilizaremos la ponderación de subclasificación para controlar estos factores.")
    df['AgeGroup'] = np.where(df['Age'] >= 27, 'old', 'young')
    df['Demographic'] = df['Sex'] + '-' + df['AgeGroup']
    df['Demographic'].unique()
    # Initialize variables to store the total weighted effect and the total weight
    average_weighted_effect = 0
    # Total number of observations in the entire dataset
    total_obs = df.shape[0]
    # Loop through each unique value in the 'Demographic' column
    for demographic in df['Demographic'].unique():
    # Filter the DataFrame for the current demographic group
        group_df = df[df['Demographic'] == demographic]
    # Calculate the total number of observations in the group
        group_obs = group_df.shape[0]
    # Calculate the difference in means for 'Survived' based on 'FirstClass'
        diff = group_df[group_df['FirstClass'] == 1]['Survived'].mean() - group_df[group_df['FirstClass'] == 0]['Survived'].mean()
    # Calculate the weight as the proportion of total observations that are in this group
        weight = group_obs / total_obs
    # Compute the weighted effect
        weighted_effect = diff * weight
    # Add the weighted effect to the total weighted effect
        average_weighted_effect = average_weighted_effect + weighted_effect
    
    st.write('Weighted Average Effect: {:.2f}%'.format(average_weighted_effect * 100))

    st.write("La ponderación por las variables género y edad (ATE ponderado), ir en primera clase tiene asociada una probabilidad de supervivencia menor.")

elif page == 'Conclusiones':
    st.title("Conclusiones")
    st.write("--> El dataset contenía valores nulos destacables en la columna Cabin, que viene de que los pasajeros que no iban en primera clase no tenían asignada una cabina.")
    st.write("--> La edad de los pasajeros se distribuye de forma asimétrica, con una mediana de 28 años.")
    st.write("--> La mayoría de los pasajeros eran hombres, pero aun así sobrevivieron más mujeres en términos porcentuales.")
    st.write("--> La mayoría de los pasajeros eran de tercera clase, pero sobrevivieron más pasajeros de primera clase en términos porcentuales.")
    st.write("--> Ser mujer o niño, así como ir en primera clase, ha favorecido la supervivencia de los pasajeros del Titanic.")
    st.write("--> La ponderación de subclasificación ha permitido controlar los factores observables de edad y sexo, obteniendo una estimación menos sesgada del ATE; es decir, la probabilidad de sobrevivir yendo en primera clase y controlando por sexo y edad es más fiable, y aun así alta (25,82%).")
    st.write("--> Estos resultados son consistentes con la historia del Titanic, donde las mujeres y los niños tenían prioridad para subir a los botes salvavidas, y los pasajeros de primera clase tenían asignadas cabinas con todo tipo de lujos.")
