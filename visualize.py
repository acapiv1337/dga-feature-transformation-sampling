import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go

def barplot(y):
    class_counts = pd.Series(y, index=y.index)

    # Create bar plot
    bar_plot_counts = class_counts.plot(kind='bar', color='skyblue')
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Instances')

    # Add text annotations for class counts
    for bar in bar_plot_counts.patches:
        bar_plot_counts.text(bar.get_x() + bar.get_width() / 2, 
                            bar.get_height() - bar.get_height() * 0.5,  # Adjust position slightly below top
                            str(int(bar.get_height())), 
                            ha='center', va='center', fontsize=10, color='black')

    # Display the plot
    plt.tight_layout()
    plt.show()

def histplot(df, col_x):
    fig = px.histogram(
    df,
    x=col_x,  # Column to plot
    nbins=20,      # Number of bins
    title="Histogram of Column1",
    labels={"Column1": "Values"},  # Axis labels
    # color="Category"  # Optional: Different colors for categories
    )

    # Update layout for better visualization
    fig.update_layout(
        xaxis_title="Values",
        yaxis_title="Frequency",
        bargap=0.1  # Gap between bars
    )

    fig.show()

def pairplot(df_resample, number='', transform='', resample='', save=False):
    counts = df_resample['act'].value_counts()
    pairplot = sns.pairplot(df_resample, hue='act', palette='bright')
    legend = pairplot._legend
    legend.texts[0].set_text(f'1 - {counts[[1]].values[0]}')
    legend.texts[1].set_text(f'2 - {counts[[2]].values[0]}')
    legend.texts[2].set_text(f'3 - {counts[[3]].values[0]}')
    legend.texts[3].set_text(f'4 - {counts[[4]].values[0]}')
    legend.texts[4].set_text(f'5 - {counts[[5]].values[0]}')
    legend.texts[5].set_text(f'6 - {counts[[6]].values[0]}')
    if save:
        plt.savefig(f'graph/pairplot/{transform} {resample}.png')
        plt.show()
    else:
        plt.show()

def plotly_scatter(df, x, y):
    # Create the scatter plot
    fig = px.scatter(df, x=x, y=y, color="act", labels={"act":"act"})

    class_counts = df['act'].value_counts()
    class_labels = {cls: f"{cls} - {count}" for cls, count in class_counts.items()}
    for i in fig.data:
        i.name = class_labels[int(i.name)]

    # Customize the layout
    fig.update_layout({'plot_bgcolor': 'white'})
    fig.show()
    
def kdeplot(df, number='', transform='', resample='', save=False):
    sns.kdeplot(df, palette='bright')
    if save:
        plt.savefig(f'graph/kdeplot/{number} {transform} {resample}.png')
        plt.show()
    else:
        plt.show()



def count_plot(df, number='', transform='', resample='', save=False, order=False):
    plt.figure(figsize=(8, 6))
    if order:
        count_order = df['act'].value_counts().index.tolist()
        sns.countplot(df, x='act', color='#97cce8', order=count_order)
    else:
        sns.countplot(df, x='act', color='#97cce8')

    # Add counts on top of each bin
    for p in plt.gca().patches:
        height = p.get_height()
        if height > 0:  # Only label bins with counts greater than 0
            plt.text(
                p.get_x() + p.get_width() / 2, height + 2,  # Positioning the text
                f'{int(height)}',  # The count value
                ha='center',  # Horizontal alignment
                va='bottom',  # Vertical alignment
                color='black'
            )

    # Add labels and title
    plt.xlabel("Class (act)")
    plt.ylabel("Count")
    # plt.title(f"Frequency of Act ({transform} {resample})")
    plt.title("Frequency of Act")

    if save:
        plt.savefig(f'graph/count_plot/{number} {transform} {resample}.png')
        plt.show()
    else:
        plt.show()

def pie_chart(df, number='', transform='', resample='', save=False):
    # define Seaborn color palette to use 
    palette_color = sns.color_palette('bright') 
    
    df_count = df['act'].value_counts().sort_index()
    label = df_count.index
    data = df_count.values
    
    # plotting data on chart 
    plt.pie(data, labels=label, colors=palette_color, autopct='%.0f%%')
    
    # Add legend with class names and counts
    legend_labels = [f"{lbl} ({count})" for lbl, count in zip(label, data)]
    plt.legend(legend_labels, title='Class', loc='upper left', bbox_to_anchor=(1, 1))
    # plt.title(f"Pie Chart of Act ({transform} {resample})")
    plt.title("Pie Chart of Act")
    if save:
        plt.savefig(f'graph/pie_chart/{number} {transform} {resample}.png')
        plt.show()
    else:
        plt.show()