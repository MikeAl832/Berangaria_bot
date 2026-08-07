from berangaria.user_bridge.dedup import MessageDeduper


def test_first_seen_not_duplicate():
    d = MessageDeduper(ttl_seconds=60)
    assert d.seen_or_add(1, 10, now=100.0) is False


def test_second_seen_is_duplicate():
    d = MessageDeduper(ttl_seconds=60)
    assert d.seen_or_add(1, 10, now=100.0) is False
    assert d.seen_or_add(1, 10, now=101.0) is True


def test_different_message_ids_independent():
    d = MessageDeduper(ttl_seconds=60)
    assert d.seen_or_add(1, 10, now=100.0) is False
    assert d.seen_or_add(1, 11, now=100.0) is False


def test_ttl_expires():
    d = MessageDeduper(ttl_seconds=10)
    assert d.seen_or_add(1, 10, now=100.0) is False
    assert d.seen_or_add(1, 10, now=111.0) is False  # expired, treated as new
