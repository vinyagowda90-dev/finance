def get_kpis(df):

    return {
        "Total Investors": len(df),
        "Average Age": round(df["age"].mean(),1),
        "Male Investors": len(df[df["gender"]=="Male"]),
        "Female Investors": len(df[df["gender"]=="Female"])
    }
