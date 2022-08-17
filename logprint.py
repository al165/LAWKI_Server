import inspect

def logprint(*args, **kwargs):
    def get_class_from_frame(fr):
        args, _, _, value_dict = inspect.getargvalues(fr)
        # we check the first parameter for the frame function is
        # named 'self'
        if len(args) and args[0] == 'self':
            # in that case, 'self' will be referenced in value_dict
            instance = value_dict.get('self', None)
            if instance:
              # return its class
              return getattr(instance, '__class__', None)
          # return None otherwise
        return None

    frame = inspect.stack()[1][0]
    try:
        _class = get_class_from_frame(frame).__name__
        _func = frame.f_code.co_name
        print(f'[ {_class}.{_func} ]', *args, **kwargs)
    except AttributeError:
        print('[ __main__ ]', *args, **kwargs)

