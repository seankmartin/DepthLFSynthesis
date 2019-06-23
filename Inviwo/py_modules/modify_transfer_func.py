from inviwopy.glm import vec4

from random import random
import os
import colorsys

#rgba_val is [content, (rgba tf_val)]
def make_one_tf_point(rgba_val):
    rgba = rgba_val[1]
    content_string = "<pos content=\"{}\" />".format(rgba_val[0])
    rgba_string = "<rgba x=\"{}\" y=\"{}\" z=\"{}\" w=\"{}\" />".format(
        rgba[0], rgba[1], rgba[2], rgba[3]
    )
    return "\n".join((
        "<Point>",
        content_string,
        rgba_string,
        "</Point>"
    ))

def save_xml_tf(tf, out_location):
    xml_header = "\n".join((
        '<?xml version="1.0" ?>',
        '<InviwoWorkspace version="2">',
        '<maskMin content="0" />',
        '<maskMax content="1" />',
        '<type content="0" />'
    ))
    points = "\n".join(
        [make_one_tf_point(i) for i in tf]
    )
    points_string = '\n'.join((
        xml_header,
        "<Points>",
        points,
        "</Points>",
        "</InviwoWorkspace>"
    ))
    file = open(out_location, 'w')
    file.write(points_string)
    file.close()

def clamp(val, min_val=0, max_val=1):
    return max(min(val, max_val), min_val)

def random_sign():
    return -1 if random() > 0.5 else 1

def random_signed_float(range_f=1.0):
    return random() * range_f * random_sign()

def shift_pos(pos, scale=0.01, min_val=0.0, max_val=1.0):
    new_pos = clamp(pos + random_signed_float(scale), min_val, max_val)
    return new_pos

def shift_rgba_color(col, scale=0.1, rotation=0.1, opacity_scale=0.1):
    h, s, v = colorsys.rgb_to_hsv(col.r, col.g, col.b)
    h = h + rotation
    s = shift_pos(s, scale)
    v = shift_pos(v, scale)
    # Don't shift low points
    if(col.a > 0.00001):
        col.a = shift_pos(col.a, opacity_scale, min_val=0.00003, max_val=0.8)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return vec4(r, g, b, col.a)

# Don't let points cross eachother, so points ordered in position
def rectify_points(vals, point_movement=0.001):
    for i in range(len(vals) - 2):
        if vals[i+1].pos < vals[i].pos:
            vals[i+1].pos = clamp(vals[i].pos + point_movement)
        elif vals[i+1].pos > vals[i+2].pos:
            vals[i+1].pos = clamp(vals[i+2].pos - point_movement)
def modify_tf(
        ivw_tf, point_movement=0.001, first_point_scale=0.1, hue_rotation=0.1, 
        other_points_scale=0.3, opacity_scale=0.05):
    vals = ivw_tf.getValues()

    # Shift the colour and the postion of the first point by a little
    vals[0].pos = shift_pos(vals[0].pos, point_movement)
    vals[0].color = shift_rgba_color(
        vals[0].color, first_point_scale, hue_rotation, 0.0)

    # Shift the other points more
    for val in vals[1:]:
        val.pos = shift_pos(val.pos, point_movement)
        val.color = shift_rgba_color(
            val.color, scale=other_points_scale, opacity_scale=opacity_scale)

    rectify_points(vals, point_movement)
    ivw_tf.setValues(vals)
    return vals

def main():
    tf_base_dir = "GeneratedTFs"
    tf_name = "gen_tf.itf"
    out_location = os.path.join(tf_base_dir, tf_name)
    return 0

if __name__ == "__main__":
    main()

