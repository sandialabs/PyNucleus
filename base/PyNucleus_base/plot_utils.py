###################################################################################
# Copyright 2021 National Technology & Engineering Solutions of Sandia,           #
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the           #
# U.S. Government retains certain rights in this software.                        #
# If you want to use this code, please refer to the README.rst and LICENSE files. #
###################################################################################


def latexOptions(fig_width=None, fig_height=None, ratio=None,
                 fontsize=10, otherMPL=None):
    import numpy as np
    from cycler import cycler
    if fig_width is None:
        # fig_width = 6.33
        fig_width = 4.9
    if fig_height is None:
        if ratio is None:
            golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
            fig_height = fig_width*golden_mean    # height in inches
        else:
            fig_height = fig_width*ratio
    MPLconf = {
        'text.usetex': True,
        'axes.titlesize': fontsize,
        'axes.labelsize': fontsize,
        'legend.fontsize': fontsize,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'lines.linewidth': 1,
        'lines.markersize': 4,
        'text.latex.preamble': r'\usepackage{amsmath,amsfonts,amssymb,mathrsfs}',
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        # 'font.family': 'STIXGeneral',
        # 'mathtext.rm': 'Bitstream Vera Sans',
        # 'mathtext.it': 'Bitstream Vera Sans:italic',
        # 'mathtext.bf': 'Bitstream Vera Sans:bold',
        # 'font.serif': 'cm',
        'font.size': fontsize,
        'figure.figsize': [fig_width, fig_height],
        'axes.prop_cycle': cycler('color', ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF'])
    }
    if otherMPL is not None:
        MPLconf.update(otherMPL)
    return MPLconf


def latexContext(fig_width=None, fig_height=None, ratio=None,
                 fontsize=10, otherMPL=None):
    from matplotlib import rc_context
    MPLconf = latexOptions(fig_width, fig_height, ratio, fontsize, otherMPL)
    return rc_context(MPLconf)


def beamerContext(fig_width=None, fig_height=None, ratio=None,
                  fontsize=8, otherMPL=None):
    MPLconf = {'lines.markersize': 1,
               'savefig.dpi': 100*4,
               'font.family': 'sans-serif',
               'font.serif': ['DejaVu Serif',
                              'Bitstream Vera Serif',
                              'Computer Modern Roman',
                              'New Century Schoolbook',
                              'Century Schoolbook L',
                              'Utopia',
                              'ITC Bookman',
                              'Bookman',
                              'Nimbus Roman No9 L',
                              'Times New Roman',
                              'Times',
                              'Palatino',
                              'Charter',
                              'serif'],
               'patch.linewidth': 0.5}
    if otherMPL is not None:
        MPLconf.update(otherMPL)
    return latexContext(fig_width, fig_height, ratio, fontsize, MPLconf)


def posterContext(fig_width=None, fig_height=None, ratio=None,
                  fontsize=25, otherMPL=None):
    MPLconf = {'lines.markersize': 10,
               'savefig.dpi': 100*4,
               'font.family': 'serif',
               'font.serif': 'cm',
               'patch.linewidth': 0.5}
    if otherMPL is not None:
        MPLconf.update(otherMPL)
    return latexContext(fig_width, fig_height, ratio, fontsize, MPLconf)


def plot_with_latex(fun, fig_width=None, fig_height=None, ratio=None,
                    fontsize=10, otherMPL=None):
    from inspect import getargspec

    argspec = getargspec(fun)

    def new_fun(*args, **kwargs):
        kwargs_new = {}
        for i in range(len(args)):
            kwargs_new[argspec[0][i]] = args[i]
        for key in kwargs:
            if key in argspec[0][:]:
                kwargs_new[key] = kwargs[key]
        with latexContext(fig_width, fig_height, ratio, fontsize, otherMPL):
            r = fun(**kwargs_new)
        return r
    return new_fun


def plot_with_beamer(fun, fig_width=None, fig_height=None, ratio=None,
                     fontsize=8, otherMPL=None):
    from inspect import getargspec

    argspec = getargspec(fun)

    def new_fun(*args, **kwargs):
        kwargs_new = {}
        for i in range(len(args)):
            kwargs_new[argspec[0][i]] = args[i]
        for key in kwargs:
            if key in argspec[0][:]:
                kwargs_new[key] = kwargs[key]
        with beamerContext(fig_width, fig_height, ratio, fontsize, otherMPL):
            r = fun(**kwargs_new)
        return r
    return new_fun


def plot_for_poster(fun, fig_width=None, fig_height=None, ratio=None,
                    fontsize=25, otherMPL=None):
    from inspect import getargspec

    argspec = getargspec(fun)

    def new_fun(*args, **kwargs):
        kwargs_new = {}
        for i in range(len(args)):
            kwargs_new[argspec[0][i]] = args[i]
        for key in kwargs:
            if key in argspec[0][:]:
                kwargs_new[key] = kwargs[key]
        with posterContext(fig_width, fig_height, ratio, fontsize, otherMPL):
            r = fun(**kwargs_new)
        return r
    return new_fun


def plotTriangle(x, y, fac, ax=None):
    if ax is None:
        ax = plt.gca()
    dx = 0.8*(x[-2]-x[-1])
    x1 =x[-1]+dx
    y2 = y[-2]*(x[-1]/x1)**fac

    ax.plot([x[-1], x[-1], x1, x[-1]],
            [y[-2], y2, y[-2], y[-2]])
    ax.text(0.5*x[-1]+0.5*x1, y[-2], str(1), horizontalalignment='right', verticalalignment='bottom')
    ax.text(x[-1], 0.5*y[-2]+0.5*y2, str(fac), horizontalalignment='right', verticalalignment='top')


def tabulate(x, results, floatfmt=None, groups=False, **kwargs):
    import numpy as np
    from . import roc
    endl = '\n'
    ltx_endl = ' \\\\'+endl
    hline = '\\hline'+endl

    def myFmt(a, fmt):
        if isinstance(a, str):
            return a
        elif a is None:
            return ''
        else:
            return fmt.format(a)

    d = []
    expected = ['theoretical']
    grpheaders = ['']
    columnfmt = 'r'
    if groups:
        for key, vals in results:
            columnfmt += '|'
            grpheaders.append('\\multicolumn{'+str(2*len(vals))+'}{c}{'+key+'}')
            for result, expectedOrder in vals:
                r = np.concatenate((np.array([[None]]),
                                    roc(x, result))).flatten()
                d.append(np.vstack((result.flatten(), r)))
                expected += [None, expectedOrder]
                columnfmt += 'rr'
    else:
        columnfmt += '|'
        for result, expectedOrder in results:
            r = np.concatenate((np.array([[None]]),
                                roc(x, result))).flatten()
            d.append(np.vstack((result.flatten(), r)))
            expected += [None, expectedOrder]
            columnfmt += 'rr'
    d = np.vstack((x.flatten(), *d)).T

    s = ''
    s += '\\begin{tabular}{'+columnfmt+'}' + endl
    if len(grpheaders) > 1:
        s += ' & '.join(grpheaders) + ltx_endl
    s += ' & '.join(kwargs['headers']) + ltx_endl
    s += hline
    for i in range(d.shape[0]):
        s += ' & '.join([floatfmt[j].format(d[i, j]) if d[i, j] is not None else ''  for j in range(d.shape[1])]) + ltx_endl
    s += hline
    s += ' & '.join([myFmt(expected[j], floatfmt[j]) for j in range(len(expected))]) + ltx_endl
    s += '\\end{tabular}'+endl
    return s


def latexFormatRate(r, digits=2):
    import numpy as np
    if abs(r-1.0) < 1e-9:
        return ''
    elif abs(r-np.around(r))<1e-9:
        return '^{{{}}}'.format(int(np.around(r)))
    else:
        return ('^{{{:.' + str(digits) + '}}}').format(r)
