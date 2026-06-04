import plotly.express as px

def gender_chart(df):

    return px.pie(
        df,
        names="gender",
        title="Gender Distribution"
    )

def age_chart(df):

    return px.histogram(
        df,
        x="age",
        nbins=10,
        title="Age Distribution"
    )

def objective_chart(df):

    return px.bar(
        df["Objective"].value_counts().reset_index(),
        x="Objective",
        y="count",
        title="Investment Objectives"
    )

def avenue_chart(df):

    return px.bar(
        df["Avenue"].value_counts().reset_index(),
        x="Avenue",
        y="count",
        title="Preferred Investment Avenue"
    )
