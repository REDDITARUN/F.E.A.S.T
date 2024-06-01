import streamlit as st
def main():
    st.title("CSE 676: Deep Learning")
    st.header("EYES ON EATS: From Image to Formula")
    st.sidebar.title("EYES ON EATS: From Image to Formula")
    st.sidebar.header("Made by: Tarun and Charvi")
    activities = ["About","Ingredients to Recipe"]
    choice = st.sidebar.selectbox("Select ",activities)
    if choice == 'About':
        st.header(
            "Team: TARUN REDDI and CHARVI KUSUMA ")
        st.header("UBID: bhanucha and charviku")
    elif choice == 'Ingredients to Recipe':
        from ingredient_detect import main as ingredient_detect
        ingredient_detect()
    else: 
        st.write("Make A selection from the dropdown")

if __name__ == '__main__':
    main()


