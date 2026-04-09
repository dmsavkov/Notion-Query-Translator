from e2b import AsyncTemplate

template = (
    AsyncTemplate()
    .from_template("code-interpreter-v1")
    .set_user("root")
    .run_cmd("pip3 install requests python-dotenv")
    .set_user("user")
)