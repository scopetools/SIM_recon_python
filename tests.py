def test_package():
    import simtk
    assert hasattr(simtk, '__version__')
