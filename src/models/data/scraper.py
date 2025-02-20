import instaloader

def get_followers(username, user, password):
    L = instaloader.Instaloader()
    L.login(user, password)
    profile = instaloader.Profile.from_username(L.context, username)
    followers = [f.username for f in profile.get_followers()]
    return followers
