from matplotlib import animation


class CustomAnimation(object):
    help_x = list()
    help_y = list()
    help_z = list()

    def animate3d(self, ite, x_vals, y_vals, z_vals, l, ani):
        print(ite)
        if ite >= len(y_vals):
            self.help_x.clear()
            self.help_y.clear()
            self.help_z.clear()
            ani.frame_seq = ani.new_frame_seq()

        else:
            ydata = y_vals[ite]
            xdata = x_vals[ite]
            zdata = z_vals[ite]
            self.help_x.append(xdata)
            self.help_y.append(ydata)
            self.help_z.append(zdata)
        l.set_data_3d(self.help_x, self.help_y, self.help_z)
        return l,

    def init3d(self, x_vals, y_vals, z_vals, l):  # only required for blitting to give a clean slate.
        self.help_x.clear()
        self.help_y.clear()
        self.help_z.clear()
        ydata = y_vals[0]
        xdata = x_vals[0]
        zdata = z_vals[0]
        self.help_x.append(xdata)
        self.help_y.append(ydata)
        self.help_z.append(zdata)
        l.set_data_3d(self.help_x, self.help_y, self.help_z)
        return l,
