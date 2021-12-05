from functools import wraps
from flask import request, abort

APPKEY = "MyUltimateSecretKeyNYAHAHAHAHAHAHAHA" 
def require_appkey(view_function):
    @wraps(view_function)
    def decorated_function(*args, **kwargs):
        if request.args.get('Apikey') and request.args.get('Apikey') == APPKEY:
            return view_function(*args, **kwargs)
        else:
            abort(401)
    return decorated_function